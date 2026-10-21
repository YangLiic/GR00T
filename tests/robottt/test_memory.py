from gr00t.model.modules.ttt_memory import FastWeights, TTTContext, TTTMemory, fast_mlp, sgd_step
import torch


def test_analytic_sgd_matches_autograd_and_meta_gradient():
    torch.manual_seed(2)
    keys = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    values = torch.randn_like(keys, requires_grad=True)
    params = [
        torch.randn(*shape, dtype=torch.double, requires_grad=True)
        for shape in ((2, 4, 5), (2, 1, 5), (2, 5, 4), (2, 1, 4))
    ]
    state = FastWeights(*params)
    loss = (fast_mlp(keys, state) - values).square().mean(dim=(1, 2)).sum()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    expected = FastWeights(*(p - 0.1 * g for p, g in zip(params, grads)))
    actual = sgd_step(keys, values, state, 0.1)
    for got, want in zip(actual.tensors(), expected.tensors()):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)
    query = torch.randn_like(keys)
    got_meta = torch.autograd.grad(fast_mlp(query, actual).square().sum(), keys, retain_graph=True)[
        0
    ]
    want_meta = torch.autograd.grad(fast_mlp(query, expected).square().sum(), keys)[0]
    torch.testing.assert_close(got_meta, want_meta, atol=1e-8, rtol=1e-8)


def test_padding_does_not_write_and_batch_members_do_not_share_weights():
    memory = TTTMemory(4, 6, 0)
    x = torch.randn(2, 3, 4)
    initial = memory.initial_state(2, x.device)
    _, updated = memory(x, initial, torch.zeros(2, dtype=torch.long), torch.tensor([True, False]))
    for before, after in zip(initial.tensors(), updated.tensors()):
        torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
    assert not torch.equal(initial.w2[0], updated.w2[0])


def test_update_then_read_and_outer_gradients():
    memory = TTTMemory(4, 6, 0)
    x = torch.randn(1, 3, 4)
    out, state = memory(x, None, torch.zeros(1, dtype=torch.long))
    out.square().mean().backward()
    for parameter in (
        memory.q_proj.weight,
        memory.k_proj.weight,
        memory.v_proj.weight,
        memory.w1_init,
        memory.w2_init,
        memory.lr_multiplier_raw,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert state.w1.grad_fn is not None


def test_inference_mode_and_state_not_in_checkpoint():
    memory = TTTMemory(4, 6, 0)
    context = TTTContext(1, 1, torch.device("cpu"))
    keys_before = set(memory.state_dict())
    with torch.inference_mode():
        result = context.apply(memory, torch.randn(1, 3, 4))
        context.advance(3)
    assert result.shape == (1, 3, 4)
    assert context.offsets.item() == 3
    assert set(memory.state_dict()) == keys_before


def test_tbptt_carries_values_but_detaches_graph():
    memory = TTTMemory(4, 6, 0)
    context = TTTContext(1, 1, torch.device("cpu"))
    context.apply(memory, torch.randn(1, 3, 4)).sum().backward()
    before = context.states[0].w1.detach().clone()
    context.detach()
    torch.testing.assert_close(context.states[0].w1, before)
    assert context.states[0].w1.grad_fn is None
    memory.zero_grad()
    context.apply(memory, torch.randn(1, 3, 4)).sum().backward()
    assert memory.w1_init.grad is None  # W0 receives gradients only through segment 1
    assert memory.k_proj.weight.grad is not None
