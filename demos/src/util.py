def is_prime(x):
    if type(x) is int and x>=2:
        for i in range(2, x-1):
            if x % i == 0:
                return False
    else:
        raise ValueError("Input must be postive integer >=2 ")
    return True
    
if __name__ == "__main__":
    n = 101
    print(n, is_prime(n))