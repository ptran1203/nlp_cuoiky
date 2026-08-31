import subprocess
import os
import torch
import pickle
import atexit
import sys

from utils.wsl_bridge import is_windows, wsl_command

class reg_model:
    def __init__(self, executable_path: str = './dist/model_exe'):
        self.executable_path = executable_path
        self.process = None

    def start(self):
        """启动进程"""
        if self.process is None or self.process.poll() is not None:
            print('Starting process...', file=sys.stderr, flush=True)
            # dist/model_exe is a Linux ELF executable - no Windows build exists, so on Windows
            # this launches it inside WSL2 instead (see utils/wsl_bridge.py). `wsl.exe` proxies
            # stdin/stdout/stderr transparently, so __call__'s pipe protocol below is unchanged.
            cmd = wsl_command(self.executable_path) if is_windows() else [self.executable_path]

            # Force this subprocess to see no GPU at all, regardless of platform - same reasoning
            # as det_wrapper.py: the frozen binary bundles its own CUDA runtime at build time, and
            # on a machine where a real GPU IS visible (e.g. Colab, unlike WSL2 which has none at
            # all) a version/driver mismatch there can hang or crash it before it's usable. Hiding
            # the GPU reproduces the one config actually proven to work.
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f'Process started with PID {self.process.pid}', file=sys.stderr, flush=True)

    def __call__(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Process is not running")

        # The subprocess always runs with CUDA_VISIBLE_DEVICES='' now (see start()) - force CPU
        # regardless of what device the caller passed, same reasoning as det_wrapper.py. Also
        # force float32 defensively (same "Input type (c10::Half) and bias type (float) should
        # be the same" failure mode confirmed in det_wrapper.py's path) in case a caller ever
        # passes a half-precision tensor here too.
        device = 'cpu'

        try:
            input_data = {
                'tensor': x.float().to(device),
                'device': device
            }
            
            print('Serializing input data...', file=sys.stderr, flush=True)
            data = pickle.dumps(input_data)
            
            # 先发送数据长度
            print('Sending data length...', file=sys.stderr, flush=True)
            self.process.stdin.write(len(data).to_bytes(4, byteorder='big'))
            
            # 发送数据
            print('Sending data...', file=sys.stderr, flush=True)
            self.process.stdin.write(data)
            self.process.stdin.flush()
            
            # 读取响应长度
            print('Reading response length...', file=sys.stderr, flush=True)
            length_bytes = self.process.stdout.read(4)
            if not length_bytes:
                raise EOFError("Process terminated unexpectedly")
            length = int.from_bytes(length_bytes, byteorder='big')
            
            # 读取响应数据
            print(f'Reading response data ({length} bytes)...', file=sys.stderr, flush=True)
            output_data = self.process.stdout.read(length)
            output = pickle.loads(output_data)
            print('Response received', file=sys.stderr, flush=True)
            
            return output['tensor']
            
        except Exception as e:
            print(f"Error during communication: {e}", file=sys.stderr, flush=True)
            if self.process.poll() is not None:
                print(f"Process terminated with code {self.process.poll()}", 
                      file=sys.stderr, flush=True)
                error = self.process.stderr.read()
                if error:
                    print(f"Process error output: {error.decode()}", 
                          file=sys.stderr, flush=True)
            raise
    
    def cleanup(self):
        """清理资源"""
        if self.process is not None:
            print('Cleaning up process...', file=sys.stderr, flush=True)
            if self.process.poll() is None:  # 如果进程还在运行
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    print('Process not responding, terminating...', 
                          file=sys.stderr, flush=True)
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        print('Process still not responding, killing...', 
                              file=sys.stderr, flush=True)
                        self.process.kill()
                        self.process.wait()
            
            self.process.stdout.close()
            self.process.stderr.close()
            self.process = None
            print('Process cleaned up', file=sys.stderr, flush=True)

    def __del__(self):
        """析构函数"""
        self.cleanup()