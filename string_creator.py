# Script that cerates the string for the instance file.
# Fill required N (number of arms) and fill the output into the instance file

N = 100
arms = [f"a{i}" for i in range(1, N + 1)]

print("arm : {" + ", ".join(arms) + "};")
print("non-fluents {")
print(f"    NUMBER_OF_ARMS = {N}.0;")
for i, a in enumerate(arms, 1):
    print(f"    ARM_NUM({a}) = {i}.0;")
print("};")