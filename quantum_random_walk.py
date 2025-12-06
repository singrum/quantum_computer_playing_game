import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv, jv


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


def plot_quantum_walk_evolution():
    # 그래프 설정
    x_min, x_max = -60, 60
    x_values = np.arange(x_min, x_max + 1)

    # 시뮬레이션 할 시간 t 목록
    time_steps = [0, 10, 30, 50]

    plt.figure(figsize=(14, 10))

    # 1. 양자 워크 시간별 변화 (Quantum Walk Evolution)
    plt.subplot(2, 1, 1)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(time_steps)))

    for i, t in enumerate(time_steps):
        # 시뮬레이션 (FFT)
        sim_probs = run_quantum_simulation_fft(t, x_min, x_max)

        # 정확한 해 (Bessel J) - 검증용
        exact_probs = calculate_quantum_exact_prob(t, x_values)

        label_text = f't = {t}'
        plt.plot(x_values, sim_probs, 'o',
                 color=colors[i], markersize=4, label=f'{label_text} (Sim)', alpha=0.7)
        plt.plot(x_values, exact_probs, '-',
                 color=colors[i], linewidth=1, alpha=0.5)

    plt.title('Quantum Walk Evolution: Probability Distribution P_t(x)', fontsize=14)
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)

    # 2. 고전 vs 양자 비교 (Classical vs Quantum at t=50)
    plt.subplot(2, 1, 2)
    t_compare = 50

    # 양자 (Quantum)
    q_probs = calculate_quantum_exact_prob(t_compare, x_values)
    plt.plot(x_values, q_probs, '-', color='purple',
             label=f'Quantum Walk (t={t_compare})', linewidth=2)
    plt.fill_between(x_values, q_probs, color='purple', alpha=0.1)

    # 고전 (Classical)
    c_probs = calculate_classical_exact_prob(t_compare, x_values)
    plt.plot(x_values, c_probs, '--', color='green',
             label=f'Classical Walk (t={t_compare})', linewidth=2)
    plt.fill_between(x_values, c_probs, color='green', alpha=0.1)

    plt.title(
        f'Comparison: Quantum (Ballistic) vs Classical (Diffusive) at t={t_compare}', fontsize=14)
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)

    # 설명 텍스트
    plt.figtext(0.5, 0.02,
                "Quantum Walk spreads linearly [-t, t] (peaks at edges). Classical Walk spreads as sqrt(t) (peak at center).",
                ha="center", fontsize=11, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    plot_quantum_walk_evolution()
