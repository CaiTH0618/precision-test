from util.case_util import *
from util.metrics_util import *
import argparse


def look_case(op: str, dtype: str, device: str, case_id: int):
    if op == "softmax":
        from cases.softmax.case import Result, Case
    elif op == "layernorm":
        from cases.layernorm.case import Result, Case
    elif op == "crossentropy":
        from cases.crossentropy.case import Result, Case
    else:
        assert False

    case_list = load_all_cases_by_name(Case, op, dtype)
    result_list = load_all_results_by_name(Result, op, dtype, device)
    baseline_result_list = load_all_results_by_name(Result, op, dtype, "baseline")

    case = case_list[case_id]
    result = result_list[case_id]
    baseline_result = baseline_result_list[case_id]

    del case_list
    del result_list
    del baseline_result_list

    # print("\nInput Tensors:")
    # for tensor in case.get_tensor_list():
    #     print(tensor)

    # print("\nBaseline Result Tensors:")
    # for tensor in baseline_result.get_tensor_list():
    #     print(tensor)

    # print("\nResult Tensors:")
    # for tensor in result.get_tensor_list():
    #     print(tensor)

    tensor_list = result.get_tensor_list()
    baseline_tensor_list = baseline_result.get_tensor_list()

    n_zero = 0
    n_inf = 0
    n_nan = 0
    n_zero_bl = 0
    n_inf_bl = 0
    n_nan_bl = 0
    max_abs_error = 0.0
    max_rel_error = 0.0
    avg_abs_error = 0.0
    avg_rel_error = 0.0
    for (tensor, baseline_tensor) in zip(tensor_list, baseline_tensor_list):
        n_zero += count_zero(tensor)
        n_inf += count_inf(tensor)
        n_nan += count_nan(tensor)
        n_zero_bl += count_zero(baseline_tensor)
        n_inf_bl += count_inf(baseline_tensor)
        n_nan_bl += count_nan(baseline_tensor)
        max_abs_error = max(max_abs_error, get_max_abs_error(tensor, baseline_tensor))
        max_rel_error = max(max_rel_error, get_max_rel_error(tensor, baseline_tensor))
        avg_abs_error += get_avg_abs_error(tensor, baseline_tensor)
        avg_rel_error += get_avg_rel_error(tensor, baseline_tensor)

        rel_errors = get_rel_error(tensor, baseline_tensor)

        # Find indices and values of max relative error
        max_rel_idx = torch.argmax(rel_errors)
        max_rel_idx_tuple = torch.unravel_index(max_rel_idx, tensor.shape)
        print(f"\nMax Rel Error Location: {max_rel_idx_tuple} (Shape={tensor.shape})")
        print(f"  Result value: {tensor[max_rel_idx_tuple]}")
        print(f"  Baseline value: {baseline_tensor[max_rel_idx_tuple]}")
        print(f"  Relative error: {rel_errors[max_rel_idx_tuple]}")

        # Look at specific indices
        indices_list = [
            (0, 51, 1),
            (0, 92, 1)
        ]
        for i, indices in enumerate(indices_list):
            print(f"\nLook at location {i}: {indices} (Shape={tensor.shape})")
            print(f"  Result value: {tensor[indices]}")
            print(f"  Baseline value: {baseline_tensor[indices]}")
            print(f"  Relative error: {rel_errors[indices]}")

    avg_abs_error /= len(tensor_list)
    avg_rel_error /= len(tensor_list)

    print("\nMetrics:")
    print(f"  Result:   Inf={n_inf}, NaN={n_nan}, Zero={n_zero}")
    print(f"  Baseline: Inf={n_inf_bl}, NaN={n_nan_bl}, Zero={n_zero_bl}")
    print(f"  Max Rel Error: {max_rel_error}")
    print(f"  Avg Rel Error: {avg_rel_error}")
    print(f"  Max Abs Error: {max_abs_error}")
    print(f"  Avg Abs Error: {avg_abs_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, required=True, help="Operator name")
    parser.add_argument("--dtype", type=str, required=True, help="Data type")
    parser.add_argument("--device", type=str, required=True, help="Device")
    parser.add_argument("--case", type=int, required=True, help="Case ID to look at")
    args = parser.parse_args()

    look_case(args.op, args.dtype, args.device, args.case)
