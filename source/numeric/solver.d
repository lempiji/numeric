module numeric.solver;

private import numeric.cost;
private import numeric.autodiff;
private import numeric.linesearch;

private import std.math: sqrt;

/**
 *L-BFGS Solver
 */
class SimpleSolver(T, size_t NInput, size_t NLBFGS = 6)
{
    alias Cost = CostFunction!(T, NInput);
    alias Options = SolverOptions!T;
    alias Result = SolverResult!T;

public:
    this(Options options = Options.init)
    {
        _options = options;
        _searcher = new BackTrackLineSearcher!(T, NInput);
    }

public:
    ref Options options() @safe @nogc pure nothrow
    {
        return _options;
    }

public:
    void setAutoDiffCost(TFunc)(TFunc fn)
    {
        _cost = new AutoDiffCostFunction!(TFunc, T, NInput)(fn);
    }
    void setNumericDiffCost(TFunc)(TFunc fn)
    {
        _cost = new NumericDiffCostFunction!(TFunc, T, NInput)(fn);
    }
    void setCostFunction(Cost cost)
    {
        _cost = cost;
    }

public:
    Result solve(T[] x)
    {
        Result result;
        result.success = false;

        _searcher.options = _options.linesearch;
        _searcher.setCostFunction(_cost);

        //current
        auto xc = x.dup;
        auto gc = new T[NInput];
        //prev
        auto xp = new T[NInput];
        auto gp = new T[NInput];
        //search vector
        auto sv = new T[NInput];

        //L-BFGS
        static if (NLBFGS > 0)
        {
            static struct LBFGSIterateData
            {
                T alpha;
                T[] y;
                T[] s;
                T iys;
            }

            auto buf = new LBFGSIterateData[NLBFGS];
            auto bufPos = 0;
            foreach (ref d; buf)
            {
                d.alpha = 0;
                d.s = new T[NInput];
                d.y = new T[NInput];
                d.iys = 0;
            }
        }

        auto fx = _cost.evaluate(xc, gc);
        result.firstCost = fx;

        T xnorm = 0;
        T gnorm = 0;
        foreach (j; 0 .. NInput)
        {
            xnorm += xc[j] * xc[j];
            gnorm += gc[j] * gc[j];
        }

        if (xnorm < 1) xnorm = 1;
        if (gnorm < xnorm * _options.gradientTolerance)
        {
            //already minimized
            result.success = true;
            result.finalCost = fx;
            return result;
        }

        //H_0 is identity matrix
        sv[] = -gc[];

        //for linesearch
        T step = _options.estimateStepSize
            ? 1.0 / sqrt(gnorm)
            : _options.initialStepSize;

        auto loop = 1;
        for (;;)
        {
            SolverIteration!T iteration;
            scope(exit) { result.iterations ~= iteration; }

            //store
            xp[] = xc[];
            gp[] = gc[];

            //linesearch
            auto lr = _searcher.search(xp, gp, sv, fx, xc, gc, step);
            fx = lr.finalCost;
            iteration.lineSearchIterations = lr.numIterations;
            iteration.success = lr.success;
            iteration.cost = lr.finalCost;
            iteration.stepSize = lr.stepSize;
            if (!lr.success)
            {
                //linesearch failed
                result.success = false;
                //restore
                xc[] = xp[];
                gc[] = gp[];
                break;
            }

            //check the gradient
            xnorm = 0;
            gnorm = 0;
            foreach (j; 0 .. NInput)
            {
                xnorm += xc[j] * xc[j];
                gnorm += gc[j] * gc[j];
            }
            iteration.paramNorm = xnorm;
            iteration.gradientNorm = gnorm;

            if (xnorm < 1) xnorm = 1;
            if (gnorm < xnorm * _options.gradientTolerance)
            {
                //convergence
                result.success = true;
                break;
            }

            if (loop >= _options.maxIterations)
            {
                //iterations is over
                result.success = false;
                break;
            }

            static if (NLBFGS > 0)
            {
                //update
                buf[bufPos].y[] = xc[] - xp[];
                buf[bufPos].s[] = gc[] - gp[];

                auto ys = dotProduct(buf[bufPos].y, buf[bufPos].s);
                if (ys == 0)
                {
                    //is the problem hard?
                    result.success = false;
                    break;
                }

                buf[bufPos].iys = 1.0 / ys;
                auto iyy = ys / dotProduct(buf[bufPos].y, buf[bufPos].y);
            }

            //compute the search vector
            sv[] = -gc[];
            static if (NLBFGS > 0)
            {
                //L-BFGS
                import std.algorithm : min;
                immutable bound = min(loop, NLBFGS);
                auto j = bufPos = (bufPos + 1) % NLBFGS;

                foreach (_; 0 .. bound)
                {
                    j = (j + NLBFGS - 1) % NLBFGS;
                    buf[j].alpha = dotProduct(buf[j].s, sv) * buf[j].iys;
                    sv[] -= buf[j].alpha * buf[j].y[];
                }
                sv[] *= iyy;
                foreach (_; 0 .. bound)
                {
                    auto beta = dotProduct(buf[j].y, sv) * buf[j].iys;
                    sv[] += (buf[j].alpha - beta) * buf[j].s[];
                    j = (j + 1) % NLBFGS;
                }
            }

            //prepare for a next
            loop++;
            step = _options.estimateStepSize
                ? 1.0 / sqrt(dotProduct(sv, sv))
                : _options.initialStepSize;
        }
        x[] = xc[];
        result.finalCost = fx;

        return result;
    }

private:
    Cost _cost;
    BackTrackLineSearcher!(T, NInput) _searcher;
    Options _options;
}

struct SolverOptions(T)
{
    size_t maxIterations = 20;
    T gradientTolerance = 1e-10;

    bool estimateStepSize = false;
    T initialStepSize = 1;

    LineSearchOptions!T linesearch;
}
struct SolverResult(T)
{
    bool success;

    T firstCost;
    T finalCost;

    SolverIteration!T[] iterations;
}
struct SolverIteration(T)
{
    bool success;

    size_t lineSearchIterations;
    T stepSize;

    T cost;
    T paramNorm;
    T gradientNorm;
}

unittest
{
    auto solver = new SimpleSolver!(double, 3);
    solver.options.linesearch.type = LineSearchType.StrongWolfe;
    solver.options.linesearch.maxIterations = 50;
    solver.options.estimateStepSize = false;
    solver.options.initialStepSize = 0.5;
    solver.options.maxIterations = 50;

    static struct Func
    {
        T opCall(T)(in T[] x)
        {
            import numeric.math;
            auto t0 = x[0] + x[1] - 1;
            auto t1 = x[1] + x[2] + 5;
            auto t2 = x[2] + x[0] + 3;
            return square(t0) + square(t1) + square(t2);
        }
    }
    Func fn;
    solver.setAutoDiffCost(fn);

    auto x = new double[3];
    x[] = 0.5;
    auto result = solver.solve(x);

    assert(result.success);
    assert(result.iterations.length <= 50);
    assert(result.firstCost > 30);
    assert(result.finalCost < 1e-10);
}

unittest
{
    import numeric.functions;

    auto solver = new SimpleSolver!(double, 3);
    solver.options.linesearch.type = LineSearchType.StrongWolfe;
    solver.options.linesearch.maxIterations = 10;
    solver.options.estimateStepSize = true;
    solver.options.maxIterations = 50;

    RosenBrockFunction fn;
    solver.setNumericDiffCost(fn);

    auto x = new double[3];
    x[0] = -1.2;
    x[1] = 0.4;
    x[2] = -0.1;
    auto result = solver.solve(x);

    assert(!result.success);
    assert(result.iterations.length == 50);
    assert(result.firstCost > 30);
    assert(result.finalCost < 5);
}
