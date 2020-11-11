import itertools


# parser.add_argument('--l2d', default=0, type=int)
# parser.add_argument('--tanh', default=0, type=int)
# parser.add_argument('--use_max', default=1, type=int)
# parser.add_argument('--neighbor', default=1, type=int)
# parser.add_argument('--drop_rate', default=0.0, type=float)
# parser.add_argument('--targ_pool_size', default=[2,2], type=list)
# parser.add_argument('--cont_pool_size', default=[2,2], type=list)
#
# parser.add_argument('--targ_ker_num', default=[12,24], type=list) # [7,28]
# parser.add_argument('--targ_ker_size', default=[1,2], type=list)
#
# parser.add_argument('--cont_ker_num', default=[10,20], type=list) # [17,72]
# parser.add_argument('--cont_ker_size', default=[1,2], type=list)
#
# parser.add_argument('--n_fc', default=1, type=int) # 280,200,120,80


targ_ker = (
    # ([],[]),
    ([7],[1]),
    ([7],[2]),
    ([7],[3]),
    ([32],[1]),
    ([32],[2]),
    ([32],[3]),
    ([7,15],[1,2]),
    ([7,15],[2,1]),
    ([7,15],[2,2]),
    ([15,32],[1,2]),
    ([15,32],[2,1]),
    ([15,32],[2,2])
)

#
# targ_ker = (
#     # ([],[]),
#     ([7],[1]),
#     ([7],[2]),
#     ([7],[3]),
#     ([32],[1]),
#     ([32],[2]),
#     ([32],[3]),
#     ([7,15],[1,2]),
#     ([7,15],[2,1]),
#     ([7,15],[2,2]),
#     ([15,32],[1,2]),
#     ([15,32],[2,1]),
#     ([15,32],[2,2])
# )



ker = zip(targ_ker,targ_ker)
n_fc = [1,2,4]


Params = list(itertools.product(ker,n_fc))

Params.append(     [     (([20,60],[2,2]),([40,80],[2,2])),4    ]    )
Params.append(     [     (([20,60],[2,2]),([40,80],[2,2])),4    ]    )
