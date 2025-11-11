"""
从 OpenAI baselines 复制的日志记录器，以避免引入额外的基于强化学习（RL）的依赖。
原始代码链接:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

# 定义日志级别常量
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    """
    键值对（Key-Value）写入器的抽象基类。
    所有具体的键值对输出格式化器都应继承此类并实现 writekvs 方法。
    """
    def writekvs(self, kvs):
        """
        写入一个键值对字典。
        :param kvs: (dict) 要写入的键值对字典。
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    序列（Sequence）写入器的抽象基类。
    用于写入一系列字符串。
    """
    def writeseq(self, seq):
        """
        写入一个字符串序列。
        :param seq: (list or tuple) 要写入的字符串序列。
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    """
    将日志以人类可读的格式输出到文件或标准输出。
    输出格式为一个带边框的表格。
    """
    def __init__(self, filename_or_file):
        """
        初始化 HumanOutputFormat。
        :param filename_or_file: (str or file object) 输出文件名或文件对象。
                                 如果是字符串，则会创建一个新文件。
                                 如果是文件对象，则直接使用该对象。
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True  # 标记文件是自己打开的，需要自己关闭
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False # 标记文件是外部传入的，不需要自己关闭

    def writekvs(self, kvs):
        """
        将键值对字典格式化为表格并写入文件。
        """
        # 创建用于打印的字符串
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val  # 浮点数格式化
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # 找到键和值的最大宽度，用于对齐
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # 写入数据
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # 将缓冲区内容刷新到文件
        self.file.flush()

    def _truncate(self, s):
        """
        截断过长的字符串，以保证表格美观。
        """
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        """
        将一个字符串序列写入文件，元素之间用空格隔开。
        """
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # 除非是最后一个元素，否则添加空格
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        如果文件是内部打开的，则关闭它。
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    """
    将键值对以 JSON 格式写入文件，每行一个 JSON 对象。
    """
    def __init__(self, filename):
        """
        初始化 JSONOutputFormat。
        :param filename: (str) 输出的 JSON 文件名。
        """
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        """
        将键值对字典转换为 JSON 字符串并写入文件。
        """
        for k, v in sorted(kvs.items()):
            # 如果值是 numpy 类型，转换为 Python 内置 float 类型
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        """
        关闭文件。
        """
        self.file.close()


class CSVOutputFormat(KVWriter):
    """
    将键值对以 CSV 格式写入文件。
    """
    def __init__(self, filename):
        """
        初始化 CSVOutputFormat。
        :param filename: (str) 输出的 CSV 文件名。
        """
        self.file = open(filename, "w+t")
        self.keys = []  # 存储 CSV 文件的表头
        self.sep = ","  # 分隔符

    def writekvs(self, kvs):
        """
        将键值对字典作为一行写入 CSV 文件。
        如果出现新的键，会自动更新表头并重写文件以添加新列。
        """
        # 将当前行添加到历史记录中
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            # 如果有新的键，需要更新表头
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            # 写入新的表头
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            # 为已有的行补上新列的空位
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        
        # 写入当前行的数据
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        关闭文件。
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    将键值对转储为 TensorBoard 的数值格式，用于可视化。
    """

    def __init__(self, dir):
        """
        初始化 TensorBoardOutputFormat。
        :param dir: (str) 存储 TensorBoard 事件文件的目录。
        """
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        # 动态导入 tensorflow，避免不使用 TensorBoard 时也需要安装它
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        """
        将键值对写入 TensorBoard 事件文件。
        """
        def summary_val(k, v):
            # 创建一个 TensorBoard 的 Summary.Value 对象
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        """
        关闭 TensorBoard 的 EventsWriter。
        """
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    """
    根据指定的格式字符串创建并返回一个输出格式化器实例。
    :param format: (str) 格式名称，如 'stdout', 'log', 'json', 'csv', 'tensorboard'。
    :param ev_dir: (str) 日志输出目录。
    :param log_suffix: (str) 日志文件后缀，用于区分不同的运行或进程。
    :return: 一个 KVWriter 或 SeqWriter 的实例。
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API - 提供给外部调用的高级接口
# 这些函数通过 get_current() 获取全局唯一的 Logger 实例来工作。
# ================================================================


def logkv(key, val):
    """
    记录一个诊断指标的键值对。
    每次迭代为每个诊断量调用一次。
    如果多次调用，将使用最后一次的值。
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    与 logkv() 相同，但如果多次调用，则会对值进行平均。
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    记录一个键值对字典。
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    将当前迭代的所有诊断指标写入到配置的输出中（如文件、控制台）。
    """
    return get_current().dumpkvs()


def getkvs():
    """
    获取当前迭代记录的所有键值对。
    """
    return get_current().name2val


def log(*args, level=INFO):
    """
    将参数序列（无分隔符）写入控制台和输出文件。
    """
    get_current().log(*args, level=level)


def debug(*args):
    """
    记录 DEBUG 级别的日志。
    """
    log(*args, level=DEBUG)


def info(*args):
    """
    记录 INFO 级别的日志。
    """
    log(*args, level=INFO)


def warn(*args):
    """
    记录 WARN 级别的日志。
    """
    log(*args, level=WARN)


def error(*args):
    """
    记录 ERROR 级别的日志。
    """
    log(*args, level=ERROR)


def set_level(level):
    """
    在当前日志记录器上设置日志阈值。
    低于此级别的日志将不会被记录。
    """
    get_current().set_level(level)


def set_comm(comm):
    """
    为日志记录器设置 MPI 通信器，用于分布式训练中的数据聚合。
    """
    get_current().set_comm(comm)


def get_dir():
    """
    获取日志文件正在写入的目录。
    如果没有输出目录（即没有调用 configure），则返回 None。
    """
    return get_current().get_dir()


# 为了向后兼容或提供别名
record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    """
    一个上下文管理器，用于分析代码块的执行时间。
    用法:
    with profile_kv("my_code_block"):
        # ... a block of code ...
    """
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        # 将执行时间累加到 'wait_scopename' 这个键上
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    一个装饰器，用于分析函数的执行时间。
    用法:
    @profile("my_func")
    def my_func():
        # ... function code ...
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend - 后端实现
# ================================================================


def get_current():
    """
    获取当前的全局 Logger 实例。
    如果实例不存在，则会配置一个默认的日志记录器。
    """
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    """
    日志记录器的主类。
    管理日志数据、输出格式和分布式通信。
    """
    DEFAULT = None  # 一个没有输出文件的默认日志记录器。
    # 这样即使没有配置任何输出文件，仍然可以向终端打印日志。
    CURRENT = None  # 当前被全局 API 函数使用的日志记录器实例。

    def __init__(self, dir, output_formats, comm=None):
        """
        初始化 Logger。
        :param dir: (str) 日志输出目录。
        :param output_formats: (list) 输出格式化器实例的列表。
        :param comm: (MPI.Comm) MPI 通信器，用于分布式环境。
        """
        self.name2val = defaultdict(float)  # 存储当前迭代的键值对
        self.name2cnt = defaultdict(int)    # 存储用于计算平均值的计数
        self.level = INFO                   # 默认日志级别
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # --- 日志 API 的具体实现 ---
    def logkv(self, key, val):
        """
        存储一个键值对。
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        存储一个键值对，并更新其计数，用于后续计算平均值。
        """
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        """
        处理并转储当前迭代的所有键值对。
        如果在分布式环境中，会先进行数据聚合。
        """
        if self.comm is None:
            # 单机模式，直接使用本地数据
            d = self.name2val
        else:
            # 分布式模式，使用 MPI 进行加权平均
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                # 非主节点上，d 可能为空，添加一个虚拟键以避免警告
                d["dummy"] = 1
        out = d.copy()  # 复制一份用于返回，例如用于单元测试
        # 将聚合后的数据写入所有配置的输出格式
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        # 清空当前迭代的数据，为下一次迭代做准备
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        """
        记录一条序列日志，如果其级别高于或等于当前设置的级别。
        """
        if self.level <= level:
            self._do_log(args)

    # --- 配置方法 ---
    def set_level(self, level):
        """
        设置日志级别。
        """
        self.level = level

    def set_comm(self, comm):
        """
        设置 MPI 通信器。
        """
        self.comm = comm

    def get_dir(self):
        """
        返回日志目录。
        """
        return self.dir

    def close(self):
        """
        关闭所有关联的输出格式化器。
        """
        for fmt in self.output_formats:
            fmt.close()

    # --- 内部方法 ---
    def _do_log(self, args):
        """
        将序列日志写入所有配置的序列写入器。
        """
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import():
    """
    在不导入 mpi4py 的情况下获取 MPI 进程的秩（rank）。
    这样做是为了避免在导入此模块时自动调用 MPI_Init()。
    通过检查环境变量来确定 rank。
    """
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0 # 如果不是 MPI 环境，默认为 rank 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
    在不同节点上的字典之间执行加权平均。
    代码复制自 OpenAI baselines。
    Input: local_name2valcount: 一个字典，映射 key -> (value, count)
    Returns: 一个字典，映射 key -> mean
    """
    # 所有进程将自己的数据发送给主进程（rank 0）
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        # 主进程计算加权平均
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        # 非主进程返回空字典
        return {}


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    配置全局日志记录器。这是设置日志系统的主要入口点。
    :param dir: (str, optional) 日志输出目录。如果为 None，会尝试从环境变量或临时目录创建。
    :param format_strs: (list of str, optional) 要使用的输出格式列表，如 ['stdout', 'log', 'csv']。
    :param comm: (MPI.Comm, optional) MPI 通信器，用于分布式训练。
    :param log_suffix: (str, optional) 添加到日志文件名的后缀。
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        # 为不同的 MPI 进程添加不同的日志后缀
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        # 根据是否在分布式环境中设置默认的输出格式
        if rank == 0: # 主进程默认输出到控制台、日志文件和 CSV
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else: # 其他进程默认只输出到日志文件
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    # 创建并设置全局 Logger 实例
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    """
    配置一个默认的日志记录器，并将其设置为 Logger.DEFAULT。
    当第一次调用日志 API 且没有显式配置时，此函数会被调用。
    """
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    """
    将全局日志记录器重置为默认状态。
    """
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    """
    一个上下文管理器，用于在特定作用域内临时配置日志记录器。
    离开作用域后，将恢复之前的配置。
    """
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        # 恢复之前的日志记录器
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger
