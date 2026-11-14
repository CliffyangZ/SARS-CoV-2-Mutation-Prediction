# """最小真機測試：需先 pip install pennylane-qiskit，並設定 IBM_QUANTUM_TOKEN。"""

# import os
# from dotenv import load_dotenv

# load_dotenv()

# import pennylane as qml
# from qiskit_ibm_runtime import QiskitRuntimeService

# token = os.environ.get("IBM_QUANTUM_TOKEN")

# # 已存過帳號可省略 save_account；instance 警告可設 instance="open-instance" 或 "auto"
# service = QiskitRuntimeService(token=token, instance="open-instance")
# backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
# print("backend:", backend.name)

import pennylane as qml

dev = qml.device("default.qubit", wires=2)# , backend=backend)

@qml.set_shots(1024)
@qml.qnode(dev)
def circuit(theta):
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))


print(circuit(0.5))
