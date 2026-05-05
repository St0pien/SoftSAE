from sae_bench.evals.core.main import multiple_evals
from sae_list import selected_saes



output = multiple_evals(selected_saes, 10, 1, compute_featurewise_density_statistics = True,
    compute_featurewise_weight_based_metrics = True,
    exclude_special_tokens_from_reconstruction = True,
                        )
