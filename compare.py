from util.case_util import *
from util.metrics_util import *
import argparse


def compare(op: str, dtype: str, device: str):
    if op == "softmax":
        from cases.softmax.case import Result
    elif op == "layernorm":
        from cases.layernorm.case import Result
    elif op == "crossentropy":
        from cases.crossentropy.case import Result
    else:
        assert False

    result_list = load_all_results_by_name(Result, op, dtype, device)
    baseline_result_list = load_all_results_by_name(Result, op, dtype, "baseline")

    n_zero_overall = 0
    n_inf_overall = 0
    n_nan_overall = 0
    max_abs_error_overall = 0.0
    max_rel_error_overall = 0.0
    avg_abs_error_overall = 0.0
    avg_rel_error_overall = 0.0

    for i, (result, baseline_result) in enumerate(zip(result_list, baseline_result_list)):
        tensor_list = result.get_tensor_list()
        baseline_tensor_list = baseline_result.get_tensor_list()

        n_zero = 0
        n_inf = 0
        n_nan = 0
        max_abs_error = 0.0
        max_rel_error = 0.0
        avg_abs_error = 0.0
        avg_rel_error = 0.0
        for (tensor, baseline_tensor) in zip(tensor_list, baseline_tensor_list):
            n_zero += count_zero(tensor)
            n_inf += count_inf(tensor)
            n_nan += count_nan(tensor)
            max_abs_error = max(max_abs_error, get_max_abs_error(tensor, baseline_tensor))
            max_rel_error = max(max_rel_error, get_max_rel_error(tensor, baseline_tensor))
            avg_abs_error += get_avg_abs_error(tensor, baseline_tensor)
            avg_rel_error += get_avg_rel_error(tensor, baseline_tensor)
        avg_abs_error /= len(tensor_list)
        avg_rel_error /= len(tensor_list)

        n_zero_overall += n_zero
        n_inf_overall += n_inf
        n_nan_overall += n_nan
        max_abs_error_overall = max(max_abs_error_overall, max_abs_error)
        max_rel_error_overall = max(max_rel_error_overall, max_rel_error)
        avg_abs_error_overall += avg_abs_error
        avg_rel_error_overall += avg_rel_error

        star1 = "****" if (n_inf > 0 or n_nan > 0) else ""
        star2 = "****" if (max_rel_error > 1e-2) else ""

        print(f"Case {i}:")
        print(f"  Inf={n_inf}, NaN={n_nan}, Zero={n_zero} {star1}")
        print(f"  Max Rel Error: {max_rel_error} {star2}")
        print(f"  Avg Rel Error: {avg_rel_error}")
        print(f"  Max Abs Error: {max_abs_error}")
        print(f"  Avg Abs Error: {avg_abs_error}")

    avg_abs_error_overall /= len(result_list)
    avg_rel_error_overall /= len(result_list)
    
    print(f"Overall:")
    print(f"  Inf={n_inf_overall}, NaN={n_nan_overall}, Zero={n_zero_overall}")
    print(f"  Max Rel Error: {max_rel_error_overall}")
    print(f"  Avg Rel Error: {avg_rel_error_overall}")
    print(f"  Max Abs Error: {max_abs_error_overall}")
    print(f"  Avg Abs Error: {avg_abs_error_overall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, required=True, help="operator name")
    parser.add_argument("--dtype", type=str, required=True, help="data type")
    parser.add_argument("--device", type=str, required=True, help="device type")
    args = parser.parse_args()
    compare(args.op, args.dtype, args.device)
    