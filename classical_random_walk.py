import numpy as np
import matplotlib.pyplot as plt
# scipy.special.iv는 이론적 비교를 위해서만 남겨둡니다 (선택적)
from scipy.special import iv


def run_monte_carlo_simulation(t, num_particles=100000):
    """
    수식을 계산하는 대신, 실제 입자들을 시뮬레이션하여 위치를 찾습니다.

    원리 (Text Eq 15.53 기반):
    1. 시간 t 동안 각 입자가 움직일 횟수 j는 평균이 t인 포아송 분포를 따릅니다.
    2. 정해진 횟수 j번만큼 이산적 랜덤 워크(좌/우 50%)를 수행합니다.
    """
    if t == 0:
        return np.zeros(num_particles)  # 모든 입자가 0에 위치

    # 1. 각 입자가 몇 번 스텝을 밟을지 결정 (Poisson Distribution)
    # n_steps[i] = i번째 입자가 움직일 총 횟수
    n_steps = np.random.poisson(t, num_particles)

    # 2. 각 입자의 최종 위치 결정
    # 각 스텝마다 +1(오른쪽) 또는 -1(왼쪽)일 확률이 0.5입니다.
    # j번 시행에서 오른쪽으로 간 횟수(success)는 이항 분포 B(j, 0.5)를 따릅니다.
    # 위치 = (+1 * right_moves) + (-1 * left_moves)
    #      = right_moves - (n_steps - right_moves)
    #      = 2 * right_moves - n_steps
    right_moves = np.random.binomial(n_steps, 0.5)
    final_positions = 2 * right_moves - n_steps

    return final_positions


def get_distribution_from_simulation(positions, x_values):
    """
    시뮬레이션 된 입자들의 위치 목록을 확률 분포로 변환합니다.
    """
    # x_values 범위 내의 빈도수 계산
    counts = np.zeros_like(x_values, dtype=float)

    # np.unique를 사용하여 각 위치별 입자 개수를 셉니다.
    unique_pos, pos_counts = np.unique(positions, return_counts=True)

    total_particles = len(positions)

    for pos, count in zip(unique_pos, pos_counts):
        # 우리가 관심있는 x_values 범위(-30~30) 안에 있는 경우만 기록
        idx = np.where(x_values == pos)[0]
        if len(idx) > 0:
            counts[idx[0]] = count / total_particles

    return counts


def calculate_exact_prob_for_comparison(t, x_values):
    """
    시뮬레이션이 잘 되었는지 확인하기 위한 이론값 (참조용 점선)
    """
    if t == 0:
        probs = np.zeros_like(x_values, dtype=float)
        center_idx = np.where(x_values == 0)[0]
        if len(center_idx) > 0:
            probs[center_idx[0]] = 1.0
        return probs
    return np.exp(-t) * iv(np.abs(x_values), t)


def plot_random_walk_evolution():
    # 그래프 설정
    x_min, x_max = -30, 30
    x_values = np.arange(x_min, x_max + 1)

    # 시뮬레이션 할 시간 t의 목록
    time_steps = [0, 0.5, 5, 20, 100]
    num_particles = 100000  # 입자 수 (많을수록 정확함)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(time_steps)))

    print(f"Simulating {num_particles} particles per time step...")

    for i, t in enumerate(time_steps):
        # 1. 직접 시뮬레이션 수행
        simulated_positions = run_monte_carlo_simulation(t, num_particles)
        sim_probs = get_distribution_from_simulation(
            simulated_positions, x_values)

        label_text = f't = {t} (Sim)'

        # 2. 시뮬레이션 결과 그리기 (막대 그래프 또는 산점도)
        # 시뮬레이션 데이터는 동그라미(o)로 표현
        plt.plot(x_values, sim_probs, 'o', label=label_text,
                 color=colors[i], markersize=5, alpha=0.8)

        # 3. 이론값(베셀 함수)을 실선으로 그려서 시뮬레이션이 맞는지 검증 (참조용)
        # 사용자가 원하지 않으면 이 부분은 주석 처리 가능하지만, 시뮬레이션의 정확도를 보여주기 위해 얇은 실선으로 추가
        exact_probs = calculate_exact_prob_for_comparison(t, x_values)
        plt.plot(x_values, exact_probs, '-',
                 color=colors[i], alpha=0.3, linewidth=1)

    plt.title(
        f'Monte Carlo Simulation of Continuous-Time Random Walk (N={num_particles})', fontsize=15)
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Probability P_t(x)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(x_min, x_max)

    plt.figtext(0.5, 0.01,
                "Dots: Direct Simulation (Poisson steps + Random Walk). Lines: Theoretical Exact Solution.",
                ha="center", fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


if __name__ == "__main__":
    plot_random_walk_evolution()
