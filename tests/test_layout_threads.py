"""The layout session sizes its thread pool to the CPU quota, not the host."""

from babeldoc.docvision import inference


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_v2_takes_the_tightest_quota_up_the_tree(tmp_path):
    write(tmp_path / "cpu.max", "max 100000\n")
    write(tmp_path / "kubepods" / "cpu.max", "800000 100000\n")
    write(tmp_path / "kubepods" / "pod1" / "cpu.max", "600000 100000\n")
    write(tmp_path / "kubepods" / "pod1" / "ctr" / "cpu.max", "max 100000\n")
    assert inference._cgroup_v2_limit(tmp_path, "0::/kubepods/pod1/ctr\n") == 6


def test_v2_namespaced_container_reads_its_root(tmp_path):
    write(tmp_path / "cpu.max", "150000 100000\n")
    assert inference._cgroup_v2_limit(tmp_path, "0::/\n") == 1.5


def test_v2_without_quota(tmp_path):
    write(tmp_path / "user.slice" / "cpu.max", "max 100000\n")
    assert inference._cgroup_v2_limit(tmp_path, "0::/user.slice\n") is None
    assert inference._cgroup_v2_limit(tmp_path, "1:name=systemd:/\n") is None


def test_v1_quota(tmp_path):
    write(tmp_path / "cpu,cpuacct" / "cpu.cfs_quota_us", "400000\n")
    write(tmp_path / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n")
    assert inference._cgroup_v1_limit(tmp_path) == 4
    write(tmp_path / "cpu,cpuacct" / "cpu.cfs_quota_us", "-1\n")
    assert inference._cgroup_v1_limit(tmp_path) is None


def threads(monkeypatch, env=None, limit=None, usable=16, total=16):
    if env is None:
        monkeypatch.delenv(inference.LAYOUT_THREADS_ENV, raising=False)
    else:
        monkeypatch.setenv(inference.LAYOUT_THREADS_ENV, env)
    monkeypatch.setattr(inference, "cgroup_cpu_limit", lambda: limit)
    monkeypatch.setattr(inference, "_usable_cpus", lambda: usable)
    monkeypatch.setattr(inference.os, "cpu_count", lambda: total)
    return inference.layout_threads()


def test_layout_threads(monkeypatch):
    assert threads(monkeypatch, limit=6) == 6
    assert threads(monkeypatch, limit=2.5) == 2
    assert threads(monkeypatch, limit=0.5) == 1
    assert threads(monkeypatch, limit=32) == 16
    assert threads(monkeypatch, limit=6, usable=4) == 4
    # No quota: leave it to onnxruntime unless affinity narrows the CPUs.
    assert threads(monkeypatch) == 0
    assert threads(monkeypatch, usable=3) == 3
    assert threads(monkeypatch, env="3", limit=6) == 3
    assert threads(monkeypatch, env="lots", limit=6) == 6
    assert threads(monkeypatch, env="0", limit=6) == 6
