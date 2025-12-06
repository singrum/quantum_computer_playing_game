import random


def is_sorted_like(s):
    """set 출력이 0,1,2,...,n-1 순서인지 검사"""
    return list(s) == list(range(len(s)))


N = 2000

for trial in range(1_000_000):
    arr = list(range(N))
    random.shuffle(arr)
    s = set(arr)

    # set은 순서가 없으므로 repr(s)를 그대로 비교할 수 없고
    # list(s)가 내부 순서다.
    if not is_sorted_like(s):
        print("순서가 깨진 예시 발견!")
        print(list(s)[:50], "...")  # 앞부분만 출력
        break

    if trial % 10000 == 0:
        print(f"{trial}회 시도 완료...")

print("끝")
