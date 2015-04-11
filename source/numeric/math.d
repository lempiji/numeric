module numeric.math;

private static import std.math;
private static import std.numeric;
private import numeric.autodiff;

auto square(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = (x.a + x.a) * x.d[];
        y.a = x.a * x.a;
        return y;
    }
    else
        return x * x;
}
unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    auto y = Var(2.0, 1);

    auto sx = square(x);
    assert(sx.a == 1);
    assert(sx.d[0] == 2);
    assert(sx.d[1] == 0);

    auto sy = square(y);
    assert(sy.a == 4);
    assert(sy.d[0] == 0);
    assert(sy.d[1] == 4);

    auto z = 3.0;
    auto sz = square(z);
    assert(sz == 9);
}

auto dotProduct(T, U)(in T[] a, in U[] b)
{
    static if (is(T : Variable!(S, N), S, size_t N) || is(U : Variable!(W, M), W, size_t M))
    {
        import std.algorithm : min;
        auto t0 = a[0] * b[0];
        foreach (i; 1 .. min(a.length, b.length))
        {
            t0 += a[i] * b[i];
        }
        return t0;
    }
    else
        return std.numeric.dotProduct(a, b);
}
unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    auto ys = new Var[3];
    foreach (i; 0 .. xs.length)
    {
        ys[i] = xs[i] = Var(i, i);
    }
    auto z = dotProduct(xs, ys);
    assert(z.a == 5);
    assert(z.d[0] == 0);
    assert(z.d[1] == 2);
    assert(z.d[2] == 4);
}
unittest
{
    auto xs = new double[3];
    auto ys = new double[3];
    foreach (i; 0 .. xs.length)
    {
        ys[i] = xs[i] = i;
    }
    auto z = dotProduct(xs, ys);
    assert(z == 5);
}
unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    auto ys = new double[3];
    foreach (i; 0 .. xs.length)
    {
        xs[i] = Var(i, i);
        ys[i] = i;
    }
    auto z = dotProduct(xs, ys);
    assert(z.a == 5);
    assert(z.d[0] == 0);
    assert(z.d[1] == 1);
    assert(z.d[2] == 2);
}

T sum(T)(in T[] xs) @safe @nogc pure nothrow
{
    T y = xs[0];
    foreach (i; 1 .. xs.length) y += xs[i];
    return y;
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    xs[0] = Var(0.2, 0);
    xs[1] = Var(0.4, 1);
    xs[2] = Var(0.6, 2);

    auto sx = sum(xs);
    assert(std.math.approxEqual(sx.a, 1.2));
    assert(sx.d[0] == 1);
    assert(sx.d[1] == 1);
    assert(sx.d[2] == 1);

    auto ys = new double[3];
    ys[0] = 0.2;
    ys[1] = 0.4;
    ys[2] = 0.6;
    auto sy = sum(ys);
    assert(std.math.approxEqual(sy, 1.2));
}

T sumsq(T)(in T[] xs) @safe @nogc pure nothrow
{
    T y = square(xs[0]);
    foreach (i; 1 .. xs.length) y += square(xs[i]);
    return y;
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    xs[0] = Var(0.2, 0);
    xs[1] = Var(0.4, 1);
    xs[2] = Var(0.6, 2);

    auto sx = sumsq(xs);
    assert(sx.a == 0.56);
    assert(sx.d[0] == 0.4);
    assert(sx.d[1] == 0.8);
    assert(sx.d[2] == 1.2);

    auto ys = new double[3];
    ys[0] = 0.2;
    ys[1] = 0.4;
    ys[2] = 0.6;

    auto sy = sumsq(ys);
    assert(sy == 0.56);
}

T sqrt(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.sqrt(x.a);
        T y = void;
        y.d[] = x.d[] * (0.5 / t);
        y.a = t;
        return y;
    }
    else
        return std.math.sqrt(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = sqrt(x);
    auto w = sqrt(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1 / (2 * w)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T exp(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.exp(x.a);
        T y = void;
        y.d[] = t * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.exp(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = exp(x);
    auto w = exp(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], w));
    assert(std.math.approxEqual(z.d[1], 0));
}

T log(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / x.a;
        y.a = std.math.log(x.a);
        return y;
    }
    else
        return std.math.log(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = log(x);
    auto w = log(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.5));
    assert(std.math.approxEqual(z.d[1], 0));
}

T sin(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.cos(x.a) * x.d[];
        y.a = std.math.sin(x.a);
        return y;
    }
    else
        return std.math.sin(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = sin(x);
    auto w = sin(y);

    assert(std.math.approxEqual(z.a, 0.909297427));
    assert(std.math.approxEqual(z.d[0], -0.416146837));
    assert(std.math.approxEqual(z.d[1], 0));
    assert(std.math.approxEqual(w, 0.909297427));
}

T cos(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = -std.math.sin(x.a) * x.d[];
        y.a = std.math.cos(x.a);
        return y;
    }
    else
        return std.math.cos(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = cos(x);
    auto w = cos(y);

    assert(std.math.approxEqual(z.a, -0.416146837));
    assert(std.math.approxEqual(z.d[0], -0.909297427));
    assert(std.math.approxEqual(z.d[1], 0));
    assert(std.math.approxEqual(w, -0.416146837));
}

T tan(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        auto t = std.math.tan(x.a);
        y.d[] = (1 + t * t) * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.tan(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = tan(x);
    auto w = tan(y);

    assert(std.math.approxEqual(z.a, w));
    assert(std.math.approxEqual(z.d[0], 1 + w * w));
    assert(std.math.approxEqual(z.d[1], 0));
}

T sinh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.cosh(x.a) * x.d[];
        y.a = std.math.sinh(x.a);
        return y;
    }
    else
        return std.math.sinh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = sinh(x);
    auto w = sinh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], std.math.cosh(1.0)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T cosh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.sinh(x.a) * x.d[];
        y.a = std.math.cosh(x.a);
        return y;
    }
    else
        return std.math.cosh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = cosh(x);
    auto w = cosh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], std.math.sinh(1.0)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T tanh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.tanh(x.a);
        T y = void;
        y.d[] = (1 - t * t) * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.tanh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = tanh(x);
    auto w = tanh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1 - w * w));
    assert(std.math.approxEqual(z.d[1], 0));
}


T asinh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / std.math.sqrt(x.a * x.a + 1);
        y.a = std.math.asinh(x.a);
        return y;
    }
    else
        return std.math.asinh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(0.5, 0);
    double y = 0.5;

    auto z = asinh(x);
    auto w = asinh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.894427));
    assert(std.math.approxEqual(z.d[1], 0));
}

T acosh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / std.math.sqrt(x.a * x.a - 1);
        y.a = std.math.acosh(x.a);
        return y;
    }
    else
        return std.math.acosh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.5, 0);
    double y = 1.5;

    auto z = acosh(x);
    auto w = acosh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.894427));
    assert(std.math.approxEqual(z.d[1], 0));
}

T atanh(T)(in T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / (1 - x.a * x.a);
        y.a = std.math.atanh(x.a);
        return y;
    }
    else
        return std.math.atanh(x);
}
@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(0.5, 0);
    double y = 0.5;

    auto z = atanh(x);
    auto w = atanh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1.33333));
    assert(std.math.approxEqual(z.d[1], 0));
}
