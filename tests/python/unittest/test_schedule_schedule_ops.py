import tvm


def test_schedule0():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    s = tvm.create_schedule(A1.op)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_schedule1():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')

    s = tvm.create_schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule2():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: x[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + x[t, i])
    res = tvm.scan(s_init, s_update, s_state)

    assert tuple(res.shape) == (m, n)
    s = tvm.create_schedule(res.op)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[res.op.scan_axis].min.value == 1)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)

def test_auto_inline():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')
    C = tvm.placeholder((m, n), name='C')
    T1 = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='T1')
    T2 = tvm.compute((m, n), lambda i, j: T1(i, j) + C(i, j), name='T2')

    s = tvm.create_schedule(T2.op)
    tvm.schedule.AutoInlineElemWise(s)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_inline_mixed():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    A1 = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='A1')
    A2 = tvm.compute(A.shape, lambda *i: A1(*i) + 2, name='A2')
    C = tvm.compute((n,), lambda i: A2[i] + A1[i], name='C')

    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    s[A1].compute_at(s[C], xo)
    s[A2].compute_inline()
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)


def test_schedule_cache():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')
    C = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')

    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", readers=[C])
    CC = s.cache_write(C, "shared")
    s[AA].compute_at(s[CC], CC.op.axis[0])
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


if __name__ == "__main__":
    test_inline_mixed()
    test_auto_inline()
    test_schedule_scan()
    test_schedule0()
    test_schedule1()
    test_schedule2()
    test_schedule_cache()
