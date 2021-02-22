file = open('test_case_0.txt', 'rb')
lines=file.readlines()


with open('test2.txt', "w") as f:
    f.write(a[np.lexsort((a[:, 0], -a[:, 1]))])
