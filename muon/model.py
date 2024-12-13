from dataclasses import fields
from typing import Any
from typing import Dict

import chex
import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import struct
from flax.linen import partitioning as nnp

from muon.dims import Dimensions
from muon.shard import sharding_constraint

AXES = Dimensions(X="X", Y="Y", N=None)


def trunc_normal_init(stddev, dtype):
    return init.truncated_normal(stddev=stddev, dtype=dtype, lower=-3., upper=3.)


@struct.dataclass
class TransformerConfig:
    param_dtype: Any
    dtype: Any
    n_vocab: int
    n_ctx: int
    n_layer: int
    d_model: int
    d_head: int
    n_head: int
    n_kv_head: int
    rope_theta: int
    ff_mult: int
    ff_act: str
    rmsnorm_eps: float
    rmsnorm_params: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    is_train: bool

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        flt = {k: v for k, v in kwargs.items() if k in signature}
        flt.update({k: jnp.dtype(v) for k, v in flt.items() if k.endswith("_dtype")})
        return cls(**flt)


class RMSNorm(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    suffix: str

    @nn.compact
    def __call__(self, x):
        eps = jnp.array([self.cfg.rmsnorm_eps], dtype=x.dtype)
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1) + eps)
        normed = x / rms[..., None]
        if self.cfg.rmsnorm_params:
            g = self.param(
                "g_" + self.suffix,
                nn.with_partitioning(init.zeros, AXES["Y"], self.global_mesh),
                [self.cfg.d_model],
                self.cfg.param_dtype,
            )
            normed *= (g + 1.).astype(self.cfg.dtype)[None, None, ...]
        return normed


class RotaryPositionEncoding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, pos_ids):
        *_, length, width = x.shape
        positions = jnp.arange(length)
        positions = positions[..., None]  # expand along width axis

        dimensions = jnp.arange(width // 2)  # half each for sin and cos
        angular_freqs = jnp.power(self.cfg.rope_theta, -dimensions / (width // 2))
        angular_freqs = angular_freqs[None, ...]  # expand along length axis

        # expand along leading axes, such as batch, group, and query head.
        while positions.ndim < x.ndim:
            positions = positions[None, ...]
            angular_freqs = angular_freqs[None, ...]

        mesh_axes = tuple(None for _ in range(positions.ndim))
        angles = positions * angular_freqs
        angles = sharding_constraint(angles, mesh_axes, self.global_mesh)

        cos = jnp.cos(angles).astype(x.dtype)
        sin = jnp.sin(angles).astype(x.dtype)
        even, odd = jnp.split(x, 2, axis=-1)
        r_even = even * cos - odd * sin
        r_odd = even * sin + odd * cos

        mesh_axes = ("X", "Y") + tuple(None for _ in range(positions.ndim - 2))
        r_even = sharding_constraint(r_even, mesh_axes, self.global_mesh)
        r_odd = sharding_constraint(r_odd, mesh_axes, self.global_mesh)
        r = jnp.concatenate([r_even, r_odd], axis=-1)
        r = sharding_constraint(r, mesh_axes, self.global_mesh)
        chex.assert_shape(r, x.shape)
        return r


class CausalMask(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh
    infty_approx: float = 1e30

    @nn.compact
    def __call__(self, x):
        i = jnp.arange(self.cfg.n_ctx)[..., None]
        j = jnp.arange(self.cfg.n_ctx)[None, ...]
        i = sharding_constraint(i, AXES["NN"], self.global_mesh)
        j = sharding_constraint(j, AXES["NN"], self.global_mesh)
        mask = jnp.less(i, j)  # i.e., j > i, indicator masks out non-causal connections
        mask = sharding_constraint(mask, AXES["NN"], self.global_mesh)
        mask = mask[None, None, None, ...]
        mask = sharding_constraint(mask, AXES["NNNNN"], self.global_mesh)
        x = x - jnp.array([self.infty_approx], dtype=x.dtype) * mask
        x = sharding_constraint(x, AXES["XYNNN"], self.global_mesh)
        return x


def get_param(self, name, dtype, shape, mesh_axes, mesh):
    initializer = trunc_normal_init(stddev=shape[-2] ** -0.5, dtype=dtype)
    # ^ gives "wrong" stddev for w_ao tensor, since it has distinct g, h, d, m axes
    # this issue will also affect the computation of newton-schulz iterations in muon,
    # which might otherwise be performed using batched matmul over the non-leading axes.
    return self.param(
        name,
        nn.with_partitioning(initializer, mesh_axes, mesh),
        shape,
        dtype,
    )


class GroupedQueryAttention(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, state):
        bsz = x.shape[0]
        shapes = Dimensions(
            B=bsz,
            T=self.cfg.sequence_len,
            M=self.cfg.d_model,
            D=self.cfg.d_head,
            G=self.cfg.n_kv_head,  # number of GQA groups
            H=self.cfg.n_head // self.cfg.n_kv_head,  # number of query heads per group
        )
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)

        kws = dict(self=self, dtype=self.cfg.param_dtype, mesh=self.global_mesh)
        w_aq = get_param(**kws, name="w_aq", shape=shapes["GHMD"], mesh=AXES["YNXN"])
        w_ak = get_param(**kws, name="w_ak", shape=shapes["GMD"], mesh=AXES["XNN"])
        w_av = get_param(**kws, name="w_av", shape=shapes["GMD"], mesh=AXES["XNN"])
        w_ao = get_param(**kws, name="w_ao", shape=shapes["GHDM"], mesh=AXES["YNNX"])

        q = jnp.einsum("bim,ghmd->bghid", x, w_aq.astype(self.cfg.dtype))
        k = jnp.einsum("bim,gmd->bgid", x, w_ak.astype(self.cfg.dtype))
        v = jnp.einsum("bim,gmd->bgid", x, w_av.astype(self.cfg.dtype))
        q = sharding_constraint(q, AXES["XYNNN"], self.global_mesh)
        k = sharding_constraint(k, AXES["XYNN"], self.global_mesh)
        v = sharding_constraint(v, AXES["XYNN"], self.global_mesh)

        q = RotaryPositionEncoding(self.cfg, self.global_mesh)(q)
        k = RotaryPositionEncoding(self.cfg, self.global_mesh)(k)
        q = sharding_constraint(q, AXES["XYNNN"], self.global_mesh)
        k = sharding_constraint(k, AXES["XYNN"], self.global_mesh)

        qk_mult = jnp.array([self.cfg.d_head ** -0.25], dtype=self.cfg.dtype)
        s = jnp.einsum("bghid,bgjd->bghij", q * qk_mult, k * qk_mult)
        s = sharding_constraint(s, AXES["XYNNN"], self.global_mesh)

        s = CausalMask(self.cfg.sequence_len, self.global_mesh)(s)
        s = sharding_constraint(s, AXES["XYNNN"], self.global_mesh)
        p = jax.nn.softmax(s, axis=-1)
        p = sharding_constraint(p, AXES["XYNNN"], self.global_mesh)
        o = jnp.einsum("bghij,bgjd->bghid", p, v)
        o = sharding_constraint(o, AXES["XYNNN"], self.global_mesh)

        r = jnp.einsum("bghid,ghdm->bim", o, w_ao.astype(self.cfg.dtype))
        r = sharding_constraint(r, AXES["XNY"], self.global_mesh)

        return r


class PositionwiseFeedforward(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        if self.cfg.ff_act_name == "swiglu":
            d_ff_in = int(self.cfg.ff_multiple * self.cfg.d_model) * 2
            d_ff_out = d_ff_in // 2
        else:
            d_ff_in = int(self.cfg.ff_multiple * self.cfg.d_model)
            d_ff_out = d_ff_in

        shapes = Dimensions(
            B=x.shape[0],
            T=self.cfg.sequence_len,
            M=self.cfg.d_model,
            E=d_ff_in,
            F=d_ff_out,
        )
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)

        kws = dict(self=self, dtype=self.cfg.param_dtype, mesh=self.global_mesh)
        w_fi = get_param(**kws, name="w_fi", shape=shapes["ME"], mesh=AXES["XY"])
        w_fo = get_param(**kws, name="w_fo", shape=shapes["FM"], mesh=AXES["YX"])

        x = jnp.einsum("btm,me->bte", x, w_fi.astype(self.cfg.dtype))
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)

        if self.cfg.ff_act_name == "swiglu":
            # a more communication-efficient implementation of swiglu would define
            # two separate projections for xg, xf with the same sharding.
            xg, xf = jnp.split(x, 2, axis=-1)
            x = jax.nn.silu(xg) * xf
        elif self.cfg.ff_act_name == "sqrelu":
            x = jnp.square(jax.nn.relu(x))
        else:
            x = getattr(jax.nn, self.cfg.ff_act_name)(x)
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)

        x = jnp.einsum("btf,fm->btm", x, w_fo.astype(self.cfg.dtype))
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)
        return x


class TransformerBlock(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x, _):
        kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)
        x += GroupedQueryAttention(**kws)(RMSNorm(**kws, suffix="a")(x))
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)

        x += PositionwiseFeedforward(**kws)(RMSNorm(**kws, suffix="f")(x))
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)
        return x, None


class Embedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, AXES["XN"], self.global_mesh)
        shapes = Dimensions(V=self.cfg.n_vocab, M=self.cfg.d_model, I=1)

        kws = dict(self=self, dtype=self.cfg.param_dtype, mesh=self.global_mesh)
        w_e = get_param(**kws, name="w_e", shape=shapes["VIM"], mesh=AXES["NNY"])

        x = jnp.take_along_axis(
            w_e.squeeze(1).astype(self.cfg.dtype)[None, ...],  # 1VM
            x[..., None],  # BT1
            axis=1,
        )
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)
        return x


class Unembedding(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, x):
        x = sharding_constraint(x, AXES["XNY"], self.global_mesh)
        shapes = Dimensions(V=self.cfg.n_vocab, M=self.cfg.d_model)

        kws = dict(self=self, dtype=self.cfg.param_dtype, mesh=self.global_mesh)
        w_u = get_param(**kws, name="w_u", shape=shapes["MV"], mesh=AXES["YN"])

        out_dtype = self.cfg.dtype if self.cfg.is_train else self.cfg.param_dtype
        x = jnp.einsum("btm,mv->btv", x, w_u.astype(out_dtype))
        x = sharding_constraint(x, AXES["XNN"], self.global_mesh)
        return x


class Transformer(nn.Module):
    cfg: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, tokens: jax.Array) -> Dict[str, Any]:
        kws = dict(cfg=self.cfg, global_mesh=self.global_mesh)
        x = Embedding(**kws)(tokens)
        x, _ = nn.scan(
            nnp.remat(TransformerBlock),
            length=self.cfg.n_layer,
            variable_axes=dict(params=0, intermediates=0),  # use axis 0 for params,sown
            variable_broadcast=False,  # no variable sharing across layers
            split_rngs=dict(params=True),  # each layer's init shall use a distinct rng
            in_axes=0,  # use n_layer first for inputted kv cache
            out_axes=0,  # use n_layer first for outputted kv cache
            metadata_params={nn.PARTITION_NAME: None},  # no pipeline parallel
        )(**kws)(x, None)
        x = Unembedding(**kws)(RMSNorm(**kws, suffix="u")(x))
        return x
