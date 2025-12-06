import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv, jv
from scipy.linalg import expm


def run_quantum_simulation_fft(t, x_min, x_max):
    """
    텍스트의 Fourier Analysis 방식을 사용한 양자 워크 시뮬레이션 (Eq 15.58)

    과정:
    1. 초기 상태 |0> 준비
    2. FFT를 통해 운동량(k) 공간으로 변환 (Fourier basis decomposition)
    3. 시간 진화 연산자 e^(i * t * cos(k)) 적용 (Applying unitary operator)
    4. IFFT를 통해 다시 위치(x) 공간으로 변환
    """
    if t == 0:
        probs = np.zeros(x_max - x_min + 1)
        if x_min <= 0 <= x_max:
            probs[-x_min] = 1.0
        return probs

    # FFT를 위한 격자 크기 설정 (파동이 반대편으로 넘어가지 않도록 충분히 크게 설정)
    # 양자 워크는 [-t, t] 범위로 빠르게 퍼지므로 여유 공간이 필요합니다.
    grid_size = int(2 * (t + 100) + (x_max - x_min))
    grid_size = max(grid_size, 2048)  # 2의 거듭제곱 권장

    # 1. 초기 상태 설정 (x=0에 위치한 델타 함수)
    psi = np.zeros(grid_size, dtype=complex)
    origin_idx = grid_size // 2
    psi[origin_idx] = 1.0

    # 2. FFT: 위치 공간 -> 운동량(k) 공간
    psi_k = np.fft.fft(psi)

    # 3. 시간 진화 (Eigenvalues lambda_k = cos(k))
    # fftfreq는 [0, 1/(2dx), ..., -1/(2dx)] 순서를 따르므로 2pi를 곱해 k값 생성
    k_values = 2 * np.pi * np.fft.fftfreq(grid_size)

    # Evolution operator U = e^(i * H * t). Eigenvalues of H are cos(k).
    # Text Eq 15.58: e^(i * t * cos k)
    evolution_op = np.exp(1j * t * np.cos(k_values))
    psi_k_evolved = psi_k * evolution_op

    # 4. IFFT: 운동량 공간 -> 위치 공간
    psi_evolved = np.fft.ifft(psi_k_evolved)

    # 확률 분포 P(x) = |psi(x)|^2
    probs_full = np.abs(psi_evolved)**2

    # 사용자가 요청한 x_min ~ x_max 범위만 추출
    indices = np.arange(x_min, x_max + 1) + origin_idx
    valid_mask = (indices >= 0) & (indices < grid_size)

    result_probs = np.zeros_like(indices, dtype=float)
    # valid_mask가 True인 곳만 데이터 복사
    valid_indices = indices[valid_mask]
    result_probs[valid_mask] = probs_full[valid_indices]

    return result_probs


def calculate_quantum_exact_prob(t, x_values):
    """
    양자 워크의 정확한 해: Bessel Function of the First Kind (J_x(t))
    P_t(x) = |J_x(t)|^2
    """
    if t == 0:
        probs = np.zeros_like(x_values, dtype=float)
        center_idx = np.where(x_values == 0)[0]
        if len(center_idx) > 0:
            probs[center_idx[0]] = 1.0
        return probs

    # jv(order, argument) -> order=x, argument=t
    amplitudes = jv(np.abs(x_values), t)
    return amplitudes**2


def calculate_classical_exact_prob(t, x_values):
    """비교를 위한 고전적 랜덤 워크 (Modified Bessel Function I_x(t))"""
    if t == 0:
        probs = np.zeros_like(x_values, dtype=float)
        center_idx = np.where(x_values == 0)[0]
        if len(center_idx) > 0:
            probs[center_idx[0]] = 1.0
        return probs
    return np.exp(-t) * iv(np.abs(x_values), t)

# --- NAND Tree Scattering Algorithm Simulation ---


def construct_scattering_hamiltonian(size, alpha):
    """
    NAND 트리 평가를 위한 해밀토니안 생성 (Text Eq 15.60)

    Args:
        size (int): 라인 전체 크기 (홀수 권장)
        alpha (float): 원점(x=0)에서의 Self-loop 에너지 (NAND 결과에 따라 0 또는 매우 큰 값)

    Returns:
        H (ndarray): Hamiltonian Matrix
    """
    H = np.zeros((size, size))
    center = size // 2

    # 인접 행렬 (Adjacency Matrix) 설정: 이웃한 노드 간 이동 확률 1/2
    # off-diagonal elements are 1/2
    for i in range(size - 1):
        H[i, i+1] = 0.5
        H[i+1, i] = 0.5

    # 원점(x=0)에 alpha 값 추가 (Tadpole rule에 의해 축약된 트리 에너지)
    H[center, center] = alpha

    return H


def create_wave_packet(size, x0, sigma, k0):
    """
    초기 가우시안 웨이브 패킷 생성
    Args:
        size: 공간 크기
        x0: 초기 위치 중심
        sigma: 패킷의 폭
        k0: 초기 운동량 (진행 방향 결정)
    """
    x = np.arange(size) - (size // 2)
    # Gaussian envelope * Plane wave
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    # Normalize
    psi = psi / np.linalg.norm(psi)
    return psi, x


def simulate_nand_scattering():
    """
    NAND 트리의 논리값(True/False)에 따른 웨이브 패킷의 산란 시뮬레이션
    """
    # 1. 설정
    L = 200  # 라인 길이 (좌우 -100 ~ 100)
    size = 2 * L + 1
    times = [0, 40, 80, 120]  # 스냅샷 찍을 시간들

    # 웨이브 패킷 설정
    # v_group = -sin(k). 오른쪽(+x)으로 가려면 sin(k) < 0 이어야 함.
    # k = -pi/2 에서 v = 1 (최대 속도)
    k0 = -np.pi / 2
    x0 = -50     # 왼쪽에서 시작
    sigma = 10   # 패킷의 폭

    psi0, x_axis = create_wave_packet(size, x0, sigma, k0)

    # 두 가지 시나리오: True (반사) vs False (투과)
    scenarios = [
        {"name": "Tree = TRUE (Reflection)", "alpha": 1000.0},  # alpha가 크면 장벽
        {"name": "Tree = FALSE (Transmission)", "alpha": 0.0}  # alpha가 0이면 투명
    ]

    fig, axes = plt.subplots(len(times), 2, figsize=(
        14, 12), sharex=True, sharey=True)

    for col_idx, scenario in enumerate(scenarios):
        # 해밀토니안 생성
        H = construct_scattering_hamiltonian(size, scenario["alpha"])

        # 시간 진화 연산자 미리 계산 (여기서는 단순화를 위해 각 단계별 expm 사용)
        # 실제로는 exp(-iHt)이지만, Scipy expm은 수학적 행렬 지수함수이므로 -1j 곱함

        psi_t = psi0.copy()
        current_t = 0

        for row_idx, target_t in enumerate(times):
            # 시간 차이만큼 진화
            dt = target_t - current_t
            if dt > 0:
                U = expm(-1j * H * dt)
                psi_t = U @ psi_t
                current_t = target_t

            ax = axes[row_idx, col_idx]

            # 파동 함수의 실수부(Real part)를 그려서 위상 변화(Oscillation)를 보여줌 (이미지 스타일)
            ax.plot(x_axis, np.real(psi_t), color='black', linewidth=1)

            # 포락선(Envelope)을 옅게 표시
            ax.plot(x_axis, np.abs(psi_t), color='red',
                    alpha=0.3, linewidth=1, linestyle='--')

            # 원점(장벽 위치) 표시
            ax.axvline(x=0, color='blue', linestyle=':', alpha=0.5)

            if row_idx == 0:
                ax.set_title(scenario["name"], fontsize=14, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f"t = {target_t}", fontsize=12)

            ax.set_ylim(-0.25, 0.25)
            ax.grid(True, alpha=0.2)

            # 텍스트 주석
            if row_idx == len(times) - 1:
                ax.set_xlabel("Position x", fontsize=12)

    plt.suptitle(
        "Quantum Walk Scattering on NAND Tree Evaluation\n(Simulating Farhi-Goldstone-Gutmann Algorithm)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 기존 랜덤 워크 플롯 대신 NAND 산란 시뮬레이션 실행
    simulate_nand_scattering()
