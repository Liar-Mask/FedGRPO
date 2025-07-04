
# from open_r1.utils.datasets import get_dataset_this_round
from datasets import load_dataset, DatasetDict, load_from_disk
import numpy as np
from scipy.stats import dirichlet

# 加载数据集
# dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
dataset_name = '../llm_datasets/OpenR1-Math-220k'
# dataset_name = '../llm_datasets/MATH-lighteval'
distribution = 'iid'
num_clients = 5
if 'OpenR1-Math' in dataset_name:
    dataset = load_from_disk(dataset_name)
    dataset = dataset['train']

    # used for OpenR1-Math-220k to get 10% data
    # 生成随机索引
    indices = np.random.permutation(len(dataset))
    # 使用随机索引重新排序数据集
    dataset = dataset.select(indices)

    shard0, shard1 = [dataset.shard(num_shards=10, index=i) for i in range(2)]
    dataset = shard0
    dataset.save_to_disk("../llm_datasets/open_math0.1")
    shard1.save_to_disk("../llm_datasets/open_math0.1-holdout")


    types = [sample["problem_type"] for sample in dataset]  # 提取所有样本的type属性
    unique_types = list(set(types))                 # 获取所有唯一type类别
    print(f"Unique problem types: {unique_types}")

    # 查看每种 unique_type 的数量
    type_counts = {t: types.count(t) for t in unique_types}
    print("Counts for each unique problem type:")
    for t, count in type_counts.items():
        print(f"{t}: {count}")
                                
    # sys.exit(0)
    alpha = 0.1                                     # 狄利克雷分布参数（控制非均匀程度）

    # 生成每个客户端的类别分布矩阵（shape: num_clients × num_types）
    class_distributions = dirichlet.rvs([alpha]*len(unique_types), size=num_clients)

    # 创建客户端索引字典
    client_indices = {i: [] for i in range(num_clients)}

    # 按类别分配样本到客户端
    for type_idx, type_name in enumerate(unique_types):
        # 获取当前类别的所有样本索引
        type_samples = [i for i, t in enumerate(types) if t == type_name]
        
        # 根据分布划分样本
        proportions = class_distributions[:, type_idx]
        split_points = (np.cumsum(proportions) * len(type_samples)).astype(int)[:-1]
        splits = np.split(np.random.permutation(type_samples), split_points)
        
        # 分配至各客户端
        for client_idx, split in enumerate(splits):
            client_indices[client_idx].extend(split.tolist())

    # 创建子数据集字典
    sub_datasets = DatasetDict({
        f"client_{i}": dataset.select(indices) 
        for i, indices in client_indices.items()
    })

    # 保存划分结果
    # sub_datasets.save_to_disk("../llm_datasets/math_lighteval_non_iid_split-n5-0.1")
    sub_datasets.save_to_disk("../llm_datasets/open_math0.1_non_iid_split-n5-0.1")


    # 统计各客户端类别分布
    import matplotlib.pyplot as plt

    for i, client in zip(range(0,5), sub_datasets):
        client_types = [s["problem_type"] for s in sub_datasets[client]]
        type_counts = {t: client_types.count(t) for t in unique_types}
        
        plt.figure(figsize=(10,10))
        plt.bar(type_counts.keys(), type_counts.values())
        plt.title(f"{client} Type Distribution")
        plt.xticks(rotation=45)
        plt.savefig(f'../llm_datasets/open_math0.1_non_iid_split-n5-0.1/openmath0.1-0.1-n5-c{i}.pdf')
else:

    dataset = load_from_disk(dataset_name)
    dataset = dataset['train']


    if distribution =='non-iid':
        types = [sample["type"] for sample in dataset]  # 提取所有样本的type属性
        unique_types = list(set(types))                 # 获取所有唯一type类别
        num_clients = 5                                # 划分子数据集数量
        alpha = 0.1                                     # 狄利克雷分布参数（控制非均匀程度）

        # 生成每个客户端的类别分布矩阵（shape: num_clients × num_types）
        class_distributions = dirichlet.rvs([alpha]*len(unique_types), size=num_clients)

        # 创建客户端索引字典
        client_indices = {i: [] for i in range(num_clients)}

        # 按类别分配样本到客户端
        for type_idx, type_name in enumerate(unique_types):
            # 获取当前类别的所有样本索引
            type_samples = [i for i, t in enumerate(types) if t == type_name]
            
            # 根据分布划分样本
            proportions = class_distributions[:, type_idx]
            split_points = (np.cumsum(proportions) * len(type_samples)).astype(int)[:-1]
            splits = np.split(np.random.permutation(type_samples), split_points)
            
            # 分配至各客户端
            for client_idx, split in enumerate(splits):
                client_indices[client_idx].extend(split.tolist())

        # 创建子数据集字典
        sub_datasets = DatasetDict({
            f"client_{i}": dataset.select(indices) 
            for i, indices in client_indices.items()
        })
            # 统计各客户端类别分布
        import matplotlib.pyplot as plt

        for i, client in zip(range(0,5), sub_datasets):
            client_types = [s["type"] for s in sub_datasets[client]]
            type_counts = {t: client_types.count(t) for t in unique_types}
            
            plt.figure(figsize=(10,10))
            plt.bar(type_counts.keys(), type_counts.values())
            plt.title(f"{client} Type Distribution")
            plt.xticks(rotation=45)
            plt.savefig(f'figs/non-iid-c{i}.pdf')

    else:

        dataset = dataset.shuffle(seed=42)

        local_datasets = []
        for i in range(num_clients):
            local_datasets.append(dataset.shard(num_clients, i, contiguous=True))
            print(f'Dataset len of client {i} is {len(local_datasets[i])}')
        sub_datasets = DatasetDict({
            f"client_{i}": local_datasets[i]
            for i in  range(num_clients)
        })

    # 保存划分结果
    # sub_datasets.save_to_disk("../llm_datasets/math_lighteval_non_iid_split-n5-0.1")
    sub_datasets.save_to_disk(f"../llm_datasets/math_lighteval-{distribution}-n{num_clients}")






