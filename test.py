import torch
import gc


def clear_cuda2():
    device = torch.device("cuda:1")

    # 删除所有在 cuda2 上的 Tensor
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                if obj.device == device:
                    del obj
        except:
            pass

    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("CUDA 2 memory cleared.")


if __name__ == "__main__":
    clear_cuda2()
