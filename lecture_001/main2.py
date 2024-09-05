import os

# https://pytorch.org/docs/stable/profiler.html
# https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59/2
import torch
from torch.profiler import ProfilerActivity, profile

def case_default():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.square(torch.randn(10000, 10000).cuda())

    prof.export_chrome_trace("trace.json")

def case_customized():

    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=trace_handler,  # torch.profiler.tensorboard_trace_handler('trace.log'),
    ) as p:
        for _ in range(10):
            torch.square(torch.randn(10000, 10000).cuda())
            p.step()  # send a signal to the profiler that the next iteration has started

if __name__ == "__main__":
    os.system("rm -rf *.json")

    case_default()
    case_customized()

    print("Finish")
