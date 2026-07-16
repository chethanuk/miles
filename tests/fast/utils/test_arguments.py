import argparse
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    _resolve_ft_components,
    _validate_async_batch_semantics,
    get_miles_extra_args_provider,
    miles_validate_args,
)
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_custom_megatron_post_save_hook_path_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--custom-megatron-post-save-hook-path", "pkg.module.hook"] + REQUIRED_ARGS)

    assert args.custom_megatron_post_save_hook_path == "pkg.module.hook"


def test_custom_megatron_post_save_hook_path_requires_save():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--custom-megatron-post-save-hook-path", "pkg.module.hook", "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="'--save' is required when custom_megatron_post_save_hook_path is set.",
    ):
        miles_validate_args(args)


class TestResolveFtComponents:
    def test_disabled_with_no_components_returns_empty_without_warning(self, caplog) -> None:
        """use_fault_tolerance off and no ft_components yields an empty list and no warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=None)
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert not any("--ft-components is ignored" in record.message for record in caplog.records)

    def test_disabled_with_components_returns_empty_and_warns(self, caplog) -> None:
        """use_fault_tolerance off but ft_components set returns empty list and logs an ignore warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=["train"])
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert any(
            "--ft-components is ignored without --use-fault-tolerance" in record.message for record in caplog.records
        )

    def test_enabled_with_no_components_returns_default(self) -> None:
        """use_fault_tolerance on with no ft_components falls back to the default ['rollout']."""
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=None)
        result = _resolve_ft_components(args)

        assert result == ["rollout"]

    def test_enabled_with_components_returns_distinct_copy(self) -> None:
        """use_fault_tolerance on with ft_components returns an equal but distinct list copy."""
        components = ["train", "rollout"]
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=components)
        result = _resolve_ft_components(args)

        assert result == ["train", "rollout"]
        assert result is not components


def test_direct_global_batch_size_with_staleness_warns(caplog):
    # 16*4 // 16 = 4 steps per drain, num_steps_per_rollout never set.
    # The flag-keyed guard misses this; the ratio guard must not.
    args = SimpleNamespace(
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=16,
        num_steps_per_rollout=None,
        max_weight_staleness=2,
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=False,
    )
    with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
        _validate_async_batch_semantics(args)
    assert any("off-policyness" in r.message for r in caplog.records)


class TestAsyncBatchSemantics:
    def _make_args(
        self,
        *,
        num_steps_per_rollout: int | None = None,
        global_batch_size: int | None = 16,
        max_weight_staleness: int | None = 2,
        use_dynamic_global_batch_size: bool = False,
        disable_rollout_trim_samples: bool = False,
        rollout_batch_size: int = 16,
        n_samples_per_prompt: int = 4,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            global_batch_size=global_batch_size,
            num_steps_per_rollout=num_steps_per_rollout,
            max_weight_staleness=max_weight_staleness,
            use_dynamic_global_batch_size=use_dynamic_global_batch_size,
            disable_rollout_trim_samples=disable_rollout_trim_samples,
        )

    @pytest.mark.parametrize(
        "num_steps_per_rollout,global_batch_size,max_weight_staleness,use_dynamic,disable_trim,expect",
        [
            # 1: num_steps_per_rollout=4 with staleness -> WARNING
            (4, 16, 2, False, False, "warn"),
            # 2: direct global_batch_size spelling (flag unset) with staleness -> WARNING
            (None, 16, 2, False, False, "warn"),
            # 3: one step per drain via num_steps_per_rollout=1 -> silence
            (1, 64, 2, False, False, "silence"),
            # 4: one step via direct gbs=64 (16*4//64=1) -> silence
            (None, 64, 2, False, False, "silence"),
            # 5: multi-step but not async (no staleness) -> silence
            (4, 16, None, False, False, "silence"),
            # 6: dynamic gbs with staleness -> INFO, no WARNING, no "redundant"
            (None, 16, 2, True, False, "info_dynamic"),
            # 7: incompatible flag pair -> AssertionError
            (None, 16, None, True, True, "assert"),
            # 8: global_batch_size None with staleness -> silence, no crash
            (None, None, 2, False, False, "silence"),
        ],
    )
    def test_async_batch_semantics_table(
        self,
        caplog,
        num_steps_per_rollout,
        global_batch_size,
        max_weight_staleness,
        use_dynamic,
        disable_trim,
        expect,
    ):
        # When num_steps_per_rollout is set, mirror miles_validate_args derivation so
        # the helper sees a final global_batch_size (call site is after that assignment).
        if num_steps_per_rollout is not None and global_batch_size is not None:
            derived = 16 * 4 // num_steps_per_rollout
            global_batch_size = derived
        args = self._make_args(
            num_steps_per_rollout=num_steps_per_rollout,
            global_batch_size=global_batch_size,
            max_weight_staleness=max_weight_staleness,
            use_dynamic_global_batch_size=use_dynamic,
            disable_rollout_trim_samples=disable_trim,
        )
        if expect == "assert":
            with pytest.raises(AssertionError, match="--disable-rollout-trim-samples"):
                _validate_async_batch_semantics(args)
            return

        with caplog.at_level(logging.INFO, logger="miles.utils.arguments"):
            _validate_async_batch_semantics(args)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        if expect == "warn":
            assert any("off-policyness" in r.message for r in warnings)
        elif expect == "silence":
            assert not any("off-policyness" in r.message for r in warnings)
            assert not any("one optimizer step per drain" in r.message for r in infos)
        elif expect == "info_dynamic":
            assert any("one optimizer step per drain" in r.message for r in infos)
            assert not any("redundant" in r.message for r in infos)
            assert not any("off-policyness" in r.message for r in warnings)
        else:
            raise AssertionError(f"unknown expect={expect}")


def test_miles_validate_args_wires_async_batch_semantics(caplog):
    """Direct --global-batch-size spelling (num-steps-per-rollout unset) must reach the helper.

    If the call is wrongly nested inside `if args.num_steps_per_rollout is not None:`,
    this config never enters that block and the warning is never emitted.
    """
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "16",
            "--n-samples-per-prompt",
            "4",
            "--global-batch-size",
            "16",
            "--max-weight-staleness",
            "2",
            "--num-rollout",
            "1",
        ]
    )
    with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
        miles_validate_args(args)
    assert any("off-policyness" in r.message for r in caplog.records)
