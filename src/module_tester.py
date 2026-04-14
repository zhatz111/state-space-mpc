import json
import numpy as np
from pathlib import Path
from data.functions import json_to_dict
from models.time_varying_ss import TimeVaryingStateSpace


if __name__ == "__main__":
    # --- Synthetic example: 4 states, 2 inputs, 3 partitions ---
    np.random.seed(42)
    top_dir = Path().absolute()
    PARENT_FILE_PATH = top_dir / "models" / "2026-03-03 B7H4 SAM 1000L Partition Model"
    model_config = json_to_dict(Path(PARENT_FILE_PATH, "B7H4_model_parameters.json"))

    a_matrices = np.array(model_config["a_matrix"])
    b_matrices = np.array(model_config["b_matrix"])

    n_states, n_inputs, n_partitions = 6, 4, 3
    t_partitions = np.array([0.0, 4.0, 8.0])  # e.g., batch days

    # a_matrices = np.random.randn(n_partitions, n_states, n_states) * 0.5
    # b_matrices = np.random.randn(n_partitions, n_states, n_inputs) * 0.3
    a1 = 2.0
    a2 = 8.0
    b1 = 3.0
    b2 = 7.0

    # 1) Create model
    model = TimeVaryingStateSpace(t_partitions, a_matrices, b_matrices)
    print(model)
    print(f"\nA at day {a1}:\n{model.get_A(a1)}")
    print(f"\nA at day {a2}:\n{model.get_A(a2)}")
    print(f"\nB at day {b1}:\n{model.get_B(b1)}")
    print(f"\nB at day {b2}:\n{model.get_B(b2)}")
