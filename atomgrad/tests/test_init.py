import math
import numpy as np
import init as init

def test_kaiming_uniform():
    gen_array = np.random.randn(10, 1)
    kaiming_array = init.kaiming_uniform(gen_array, a=math.sqrt(5))
    print(kaiming_array.shape)

# TODO: Add more test case to compare with Pytorch
