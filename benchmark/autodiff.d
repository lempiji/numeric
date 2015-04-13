import numeric.autodiff;
import numeric.math;
import numeric.functions;

import std.stdio;
import std.datetime;
import std.random;

void main()
{
    writeln(cast(Duration)testRosenBrockFunction!(double, 100, 10000)());
}

Variable!(T, N)[] makeRandomArray(T, size_t N)()
{
    alias Var = Variable!(T, N);
    auto xs = new Var[N];
    foreach (i, ref x; xs) x = Var(uniform(-1.0, 1.0), i);
    return xs;
}

TickDuration testRosenBrockFunction(TFloat, size_t Dim, size_t Loop)()
{
    auto xs = makeRandomArray!(TFloat, 100)();
    RosenBrockFunction f;

    StopWatch sw;
    sw.start();
    foreach (_; 0 .. Loop)
    {
        auto y = f(xs);
    }
    sw.stop();
    return sw.peek();
}
