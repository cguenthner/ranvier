# Ranvier

Inspired by libraries such as [TensorFlow](https://github.com/tensorflow/tensorflow), Ranvier is a Clojure
library for numerical optimization of tensor expressions. Ranvier makes it easy to express a computation and
then find local minima using gradient-based methods with automatic differentiation ("backpropagation" in the
modern deep learning literature).

## Getting Started

Add Ranvier to your dependencies. If using leiningen, add the following to your `:dependencies` in
`project.clj`:

```
[ranvier "0.1.1"]
```

and require the `core` namespace:

```
(require '[ranvier.core :as r])
```

## API Documentation

This document provides a brief overview of Ranvier's key functionalities. Consult the [API
docs](https://cguenthner.github.io/ranvier/docs/index.html) for details. All of Ranvier's documentation
assumes a basic understanding of common numerical optimization techniques. If you do not have this background,
then Andrew Ng's Coursera classes on [machine learning](https://www.coursera.org/learn/machine-learning) and
[deep learning](https://www.coursera.org/specializations/deep-learning) are highly recommended.

## Overview

Using Ranvier generally involves two steps:

1. _Building a computation graph_, and
2. _Finding a minimum of the computation graph_

The following expression constructs a computation graph for `2 + 3`:

```
(r/mul 3 7)
;; => A data structure representing a graph
```

You'll note that the output of this function is _not_ the number `5` but rather a data structure representing
a computation graph. The exact nature of this data structure is an implementation detail. In order to actually
execute the calculation (i.e. to "run the graph" or perform "forward propagation"), you'll need to `evaluate`
it:

```
(r/evaluate (r/mul 3 7))
;; => #Tensure
;;    21.0000
```

Ranvier uses [Tensure](https://github.com/cguenthner/tensure) on the backend for performing tensor
computations, so the result of evaluating the graph is a Tensure object.

`r/mul` is an "op" (operation). This is simply a function that returns a computation graph node. Ops also
often take other computation graph nodes as arguments. In this example, Ranvier implicitly constructs nodes
for the constants `2` and `7`. The macro `G` makes it more convenient to build computation graphs. Using `G`,
the above computation can be expressed as follows:

```
(require '[ranvier.core :as r :refer [G]])
(r/evaluate (G (* 3 7)))
;; => #Tensure
;;    21.0000
```

`G` provides two advantages over calling ops directly:

1. `G` will automatically resolve any op symbols to the corresponding op in `ranvier.core`. In a few cases, it
   will resolve aliases for ops to those ops instead of to the similarly named functions in `clojure.core`: in
   the context of `G`, `*` will evaluate to `r/mul` rather than `clojure.core/*`, `+` will evaluate to `r/add`
   rather than to `clojure.core/+`, etc.
2. `G` provides some additional error handling functionality that can make it easier to debug problems that
   occur when running graphs. In particular, graph execution-time errors from graphs constructed with `G`
   will often include information about where in source code the node producing the error was constructed and
   the values of the node's operands (or of the node's gradient during backpropagation).

Ranvier allows you to construct graphs using keyword variable names and then to provide the values for those
variables later, when the graph is run:

```
(def graph (G (* :a :b)))
(def inputs {:a 3 :b 7})
(r/evaluate graph inputs)
;; => #Tensure
;;    21.0000
```

`evaluate` 1) runs forward) propagation, and 2) returns the result of the graph's root node. `r/forward`
returns the values of all nodes. Continuing the example immediately above,

```
(def node-values (r/forward graph inputs))
;; => {:a #Tensure 3.0000
;;     :b #Tensure 7.0000
;;     :mul-218502 #Tensure 21.0000}
```

Nodes that are not explicitly named (such as the root `add` node in this example) are assigned unique
names. `r/backward` runs backpropagation, taking the graph and node values from forward propagation as
arguments:

```
(r/backward graph node-values)
{:b #Tensure 3.0000,
 :a #Tensure 7.0000,
 :mul-219365 #Tensure 1.0000}
```

`r/backward` returns the gradient of the root node with respect to any other node. Thus, if we call the output
`y`, then `dy/dy = 1`, `da/dy = 7`, and `db/dy = 3`.

Consider another expression, `(a - 3)^2`:

```
(def graph (G (pow (- :a 3) 2)))
(r/simple-optimize graph {:iterations 100
                          :hyperparams {:learning-rate 0.1}})
;; => {:a #Tensure 3.0000}
```

`r/simple-optimize` runs basic gradient descent to find a minimum of the graph with respect to all unspecified
inputs.

Conceptually, that's it. Everything else is details.

## Operations

The operations available in Ranvier are summarized in the table below. This is followed by a more in-depth
discussion of a few of the less straightforward ops. There are a few general points to keep in mind:

- Since ranvier uses [Tensure](https://github.com/cguenthner/tensure) on the backend, most Ranvier ops are
  simple analogs of Tensure functions. If you have Tensure code that performs an eager computation, you can
  often convert it directly to Ranvier code for building a computation graph merely by changing the namespace
  of the functions from `tensure.core` to `ranvier.core` (and/or by wrapping the expression in `G`).
- Ranvier ops implicitly broadcast operands using the same rules as Tensure. See the [Tensure docs on
  broadcasting](https://github.com/cguenthner/tensure#broadcasting).
- We can refer to an op's "arguments" and to an op's "operands". The "operands" must be computation graph
  nodes; their values are determined at graph runtime. The "arguments" can be anything, and their values are
  determined at graph construction-time. Most arguments to ops are operand nodes: they are evaluated at
  construction-time to yield a node, which is then evaluated at run-time to some value. There are also
  non-operand arguments from which graph nodes are _not_ constructed; these typically specify some aspect of
  graph behavior that must be specified at construction- rather than run-time.
- Most op functions take as operands one or more tensors and return a tensor, but some ops accept non-tensor
  operands. This language is somewhat figurative: the op function receives operand nodes at construction
  time, but these nodes have tensor or non-tensor values at runtime. When discussing types of op inputs, it is
  thus important to distinguish between _construction-time_ and _run-time_.
- Not all operations 'backpropagate' through all operands. For example, `reshape` takes a tensor and a shape
  (as a clojure vector) and returns a new tensor. `reshape` backpropagates gradients through the tensor
  argument but not through the shape argument. That is, Ranvier can automatically compute the gradient of
  graph output with respect to the tensor argument to a `reshape` op but not with respect to the `shape`
  argument. In general, if the value of the op changes continuously with respect to the operand, it will
  backpropagate gradients through that operand. (If op output is discontinuous with respect to an operand's
  input or if the operand is in a discontinuous domain, then standard gradient-based optimization cannot be
  applied.)
- Ranvier will make inferences about the types of inputs and constants and coerce run-time values to the
  correct type (more on this [below](#type-inference)).
- Ranvier will implicitly construct nodes where operands are expected. More on this in the section on [node
  types](#node-types).

| Op or Alias (Op) | Operation                                                             |
|------------------|-----------------------------------------------------------------------|
| **Arithmetic**                                                                           |
| + (add)          | +                                                                     |
| - (sub)          | -                                                                     |
| * (mul)          | *                                                                     |
| / (div)          | /                                                                     |
| negate           | unary -                                                               |
| pow              | ^/**                                                                  |
| log              | base e logarithm (ln x)                                               |
| exp              | base e power (e^x)                                                    |
| abs              | absolute value                                                        |
| mmul             | tensor dot product                                                    |
|                                                                                          |
| **Trigonometric and hyperbolic**                                                         |
| cos              | cosine                                                                |
| sin              | sine                                                                  |
| tan              | tangents                                                              |
| acos             | arccosine                                                             |
| asin             | arcsin                                                                |
| atan             | arctangent                                                            |
| cosh             | hyperbolic cosine                                                     |
| tanh             | hyperbolic tangent                                                    |
| sinh             | hyperbolic sine                                                       |
|                                                                                          |
| **Min/max**                                                                              |
| min              | elementwise minimum                                                   |
| max              | elementwise maximum                                                   |
|                                                                                          |
| **Reducing and masking**                                                                 |
| esum             | sum of elements                                                       |
| max-along        | maximum along some dimension(s)                                      |
| min-along        | minimum along some dimension(s)                                       |
| sum-along        | sum along some dimension(s)                                           |
| max-mask         | mask of elements that are maxima along some dimension(s)              |
| min-mask         | mask of elements that are minima along some dimension(s)              |
|                                                                                          |
| **Tensor creators**                                                                      |
| zeros            | tensor filled with 0's                                                |
| ones             | tensor filled with 1's                                                |
| random-uniform   | tensor filled with numbers uniformly distributed in [0, 1)            |
| random-normal    | tensor filled with numbers normally distributed around 0 with S.D. 1  |
|                                                                                          |
| **Logical**                                                                              |
| and              | &&                                                                    |
| or               | ||                                                                    |
| not              | logical not (!/~)                                                     |
| = (eq)           | elementwise equality                                                  |
| not= (ne)        | elementwise inequality                                                |
| > (gt)           | elementwise >                                                         |
| >= (ge)          | elementwise >=                                                        |
| < (lt)           | elementwise <                                                         |
| <= (le)          | elementwise <=                                                        |
| tensor-if        | logical branching operator                                            |
|                                                                                          |
| **Joining and splitting**                                                                |
| join-along       | joins tensors along some dimension                                    |
| partition-along  | splits a tensor into chunks along some dimension (returns a seq)      |
| slices           | seq of slices through some dimension                                  |
| select-range     | subregion of a tensor                                                 |
|                                                                                          |
| **Working with shapes**                                                                  |
| reshape          | reshapes a tensor                                                     |
| size             | size (as a scalar) of a tensor along some dimension                   |
| shape            | shape of a tensor                                                     |
| make-shape       | constructs a tensor shape from scalars                                |
|                                                                                          |
| **Misc**                                                                                 |
| transpose        | tensor transpose                                                      |
| shift            | shifts elements along some dimension(s)                               |
| pad-with         | pads a tensor at the edges with some value                            |
| pad              | pads a tensor at the edges with 0                                     |
| fork             | loops over a seq of tensors                                           |
| named            | renames a node                                                        |

### Extrema ops

There are a few different sets of ops for calculating minima and maxima. `min` and `max` calculate the
_elementwise_ mins or maxes of an arbitrary number of tensors:

```
(r/evaluate (G (max [[-1 0]
                     [2 1]]
                    [[-2 7]
                     [2 0]]
                    [[-3 11]
                     [2 -1]])))
;; => #Tensure
;;    [[   -1.0000,   11.0000],
;;     [    2.0000,    1.0000]]
```

`min-along` and `max-along` compute the extrema along some dimension(s) of a single tensor.

```
(require '[tensure.core :as m])
; `t` is a 3 x 2 x 2 tensor.
(def t [[[-1 0]
         [2 1]]
        [[-2 7]
         [2 0]]
        [[-3 11]
         [2 -1]]])

; Find max along the 0th axis, and remove dimension 0
; `t` is an operand to `max-along`. The options map is a non-operand argument.
(def a (r/evaluate (G (max-along t {:axis 0
                                    :collapse true}))))
;; => #Tensure
;;    [[   -1.0000,   11.0000],
;;     [    2.0000,    1.0000]]
(m/shape a)
;; => [2 2]

; Find the max along all three axes, and keep all three dimensions
(def b (r/evaluate (G (max-along t {:axis [0 1 2]
                                    :collapse false}))))
;; => #Tensure [11.0000]
(m/shape b)
;; => [1 1 1]

; Find the largest value in the tensor. The result is a scalar.
(def c (r/evaluate (G (max-along t {:axis nil
                                    :collapse true}))))
;; => #Tensure 11.0000
(m/shape c)
;; => nil
```

Given an `axis` argument like that received by `max-along` and `min-along`, `max-mask` and `min-mask` return
"masks" of the maximum elements along the chosen axis(es). The masks are the same shapes as the input tensors;
they have 1's in the positions of the maxima and 0's everywhere else. Continuing the example immediately
above,

```
(r/evaluate (G (max-mask t 0)))
;; => #Tensure [[[    1.0000,         0],
;;               [    1.0000,    1.0000]],
;;              [[         0,         0],
;;               [    1.0000,         0]],
;;              [[         0,    1.0000],
;;               [    1.0000,         0]]]

(r/evaluate (G (max-mask t nil)))
;; => #Tensure [[[         0,         0],
;;               [         0,         0]],
;;              [[         0,         0],
;;               [         0,         0]],
;;              [[         0,    1.0000],
;;               [         0,         0]]]
```

### `report`

The `report` op takes any number of nodes followed by two functions:

- A _reporter_ function. At graph runtime, this receives the _values_ of the nodes passed to `report` and
  may return anything.
- A _logger_ function that receives the output of the reporter function.

```
(r/simple-optimize (G (report :a identity println))
                   {:iterations 5
                    :hyperparams {:learning-rate 1}})
;; Prints:
;;  #Tensure 0
;;  #Tensure -1.0000
;;  #Tensure -2.0000
;;  #Tensure -3.0000
;;  #Tensure -4.0000
```

In the example above, `report` is passed a single node, an input node named `:a`. Each time the graph runs,
`:a` will be passed to the reporter function (`identity`), and the result will be passed to the logger
function (`println`). On each iteration, the value of `:a` will be printed. `:a` is a parameter during
optimization, and, since we don't provide a starting value for it, Ranvier gives it a starting value of
`0`. Since `:a` is also the output of the graph, the gradient of graph output with respect to `:a` will
be 1. Since the learning rate is 1, and we're performing simple gradient descent, `:a` will decrease in value
by 1 on every iteration. We therefore see scalars 0, -1, -2, -3, -4 printed as gradient descent decreases the
value of `:a` and the value is reported on each iteration.

```
(let [counter (atom 0)
      logger #(->> (vec %)
                   (format "Iteration %d: %s" (swap! counter inc))
                   println)
      g (G (report :a (* :a 7) #(map m/->int %&) logger))]
  (r/simple-optimize g {:iterations 5
                        :params {:a 5}
                        :hyperparams {:learning-rate 1}}))
;; Prints:
;;  Iteration 1: [5 35]
;;  Iteration 2: [4 28]
;;  Iteration 3: [3 21]
;;  Iteration 4: [2 14]
;;  Iteration 5: [1 7]
```

This example differs from the previous one in a few ways. First, we're providing an initialization value for
`:a` (5), so optimization starts at that point. The output of the graph is still the value of `:a`: although
`report` is the graph's root node, it always passes through the value of the _first_ node it receives. Second,
the reporter function is now receiving two node values `:a` and `(* :a 7)`, converting them to integers, and
returning a seq. The logger keeps track of the iteration number and prints the seq as a vector.

This is just a toy example. In most cases, the reporter will perform some meaningful calculation: for
instance, if working on a machine learning classification problem, the reporter might calculate the percentage
of examples classified correctly. Iteration counts will also typically be reported by the the optimizer rather
than by a reporter (more on how to do this below).

### Tensor creators

Ranvier provides four ops for generating tensors: `zeros`, `ones`, `random-uniform`, and
`random-normal`. These ops _generate new tensors every time the graph is run_. If you want a tensor generated
_once_ when the graph is constructed, you can use a constant. If a graph is optimized more than once after
construction and you want a new tensor generated before every round of optimization, then provide the
generated value as an input or provide an initializer function (see the section on
[initialization](#initialization)).

```
; Using the `random-uniform` op, a new tensor is generated once on every iteration.
(r/simple-optimize (G (-> (random-uniform [2 2])
                          (report identity println)))
                   {:iterations 3})
;; Prints:
;;    #Tensure [[    0.7803,    0.1378],
;;              [    0.0259,    0.2758]]
;;    #Tensure [[    0.5042,    0.8290],
;;              [    0.0257,    0.6327]]
;;    #Tensure [[    0.8884,    0.3049],
;;              [    0.1743,    0.6263]]

; `m/sample-uniform` is a Tensure function that is run when the graph is constructed to generate a constant
; node.
(r/simple-optimize (G (-> (m/sample-uniform [2 2])
                          (report identity println)))
                   {:iterations 3})
;; Prints;
;;    #Tensure [[    0.5791,    0.1215],
;;              [    0.4834,    0.6693]]
;;    #Tensure [[    0.5791,    0.1215],
;;              [    0.4834,    0.6693]]
;;    #Tensure [[    0.5791,    0.1215],
;;              [    0.4834,    0.6693]]

; Here the graph `g` is constructed once, but it is optimized three times, with a new value for `:a` being
; provided on every round of optimization.
(let [g (G (report :a identity println))]
  (dotimes [n 3]
    (println "Round " n)
    (r/simple-optimize g
                       {:iterations 3
                        :inputs {:a (m/sample-uniform nil)}})
    (println)))
;; Prints:
;;    Round  0
;;    #Tensure 0.6791
;;    #Tensure 0.6791
;;    #Tensure 0.6791
;;
;;    Round  1
;;    #Tensure 0.0327
;;    #Tensure 0.0327
;;    #Tensure 0.0327
;;
;;    Round  2
;;    #Tensure 0.5018
;;    #Tensure 0.5018
;;    #Tensure 0.5018
```

Note that in the above examples, there are no parameters, so `simple-optimize` is not optimizing
anything. It's just running a graph multiple times in sequence.

### Logical operations and branching

In Ranvier, as in tensure, `true` is represented as a floating point `1.0` and `false` as `0.0`. There are no
true boolean tensors. Every value that is not 0.0 is 'truthy'. The `tensor-if` op provides basic branching
functionality:

```
(let [g (G (tensor-if :flag-on
                 7
                 -3))]
  (doseq [flag-value [0 0.5 1]]
    (println "flag =" flag-value ":"
             (r/evaluate g {:flag-on flag-value}))))
;; Prints:
;;    flag = 0:  #Tensure -3.0000
;;    flag = 0.5:  #Tensure 7.0000
;;    flag = 1:  #Tensure 7.0000

(let [g (G (tensor-if (>= :value :threshold)
                      7
                      -3))]
  (doseq [value [50 100 150]
          threshold [75 100 125]]
    (println "value =" value "  threshold =" threshold "  result = "
             (m/->int (r/evaluate g {:value value
                                     :threshold threshold})))))
;; Prints:
;;    value = 50   threshold = 75   result =  -3
;;    value = 50   threshold = 100   result =  -3
;;    value = 50   threshold = 125   result =  -3
;;    value = 100   threshold = 75   result =  7
;;    value = 100   threshold = 100   result =  7
;;    value = 100   threshold = 125   result =  -3
;;    value = 150   threshold = 75   result =  7
;;    value = 150   threshold = 100   result =  7
;;    value = 150   threshold = 125   result =  7

```

`tensor-if` does _not_ backpropagate gradients through its first operand: since an effectively boolean value
is in a discontinuous domain, it is not possible to optimize it using continuous methods. Otherwise, gradient
calculations with `tensor-if` work as expected:

```
(let [g (G (* :c (tensor-if :flag
                            :a
                            :b)))]
  (doseq [flag [0 1]
          :let [values (r/forward g {:flag flag
                                     :a 7
                                     :b -3
                                     :c 2})]]
    (println "\n\nWith flag =" flag "")
    (-> (r/backward g values [:a :b :c :flag])
        (ranvier.utils/update-vals m/->int)
        pprint)))
;; Prints:
;;    With flag = 0
;;    {:c -3, :b 2, :flag 0, :a 0}

;;    With flag = 1
;;    {:c 7, :a 2, :flag 0, :b 0}
```

If the output of the graph is y, then when the first branch is taken, dy/dc = 7 (the value of the first
branch), while, if the second branch is taken, dy/dc = -3 (the value of the second branch). The values of
dy/da and dy/db are 0 if the corresponding branches are not taken, as expected. But even though the value of
`:flag` influences the output, dy/dflag is 0.

For machine learning applications, one common use for `tensor-if` is to change the computation slightly based
on whether you're training a model or using it to make predictions.

### `named`

Most nodes that you'll want to refer to by name are input nodes created by using a keyword in a graph
definition. But in some cases, it can be useful to refer to other types of nodes (constant nodes or operation
nodes). In these cases, you can rename a node with `:named`.

```
(r/forward (G (* 7 3)))
;; => {:const55276 #Tensure 7.0000
;;     :const55277 #Tensure 3.0000
;;     :mul-255278 #Tensure 21.0000}

(r/forward (G (named :twenty-one (* (named :three (m/array 3))
                                    (named :seven (m/array 7))))))
;; => {:three #Tensure 3.0000
;;     :seven #Tensure 7.0000
;;     :twenty-one #Tensure 21.0000}
```

The graphs in the above examples include three nodes: two constant nodes (one for 7 and one for 3) and one op
node (for the multiplication). In the first instance, these nodes are all assigned names. In the second case,
they are named explicitly. Note also that in the first example, 7 and 3 (instances of `java.lang.Long`) are
implicitly converted into `Tensure` objects: Ranvier "knows" they must be tensors, because they are used as
arguments to `*`, which can only accept tensors. But in the second example, 7 and 3 are used as arguments to
`named`, which can accept nodes with values of any type; Ranvier is not "smart" enough to know that it needs
to convert those values to tensors, so it must be done explicitly. See the section on [type
inference](#type-inference) for additional discussion of this issue.

### `fork` (_Experimental!_)

`fork` introduces into Ranvier the concept of _recurrence_: with `fork` it is possible to define an operation
whose result feeds back into that operation. `fork` also allows you to perform simple mapping and reduction
operations similar to Clojure's `for`, `map`, `reduce`, and `loop`/`recur` constructs but with full support
for backpropagation. `fork` is necessary for implementing recurrent neural network models using
Ranvier. Although `fork` introduces many possibilities, it is currently experimental: _there are known bugs
when using nested `fork`s_, and `fork` is likely not the best abstraction for looping and recurrence. It may
be replace with alternative constructs in future versions of Ranvier.

You can think of `fork` as a combination of `clojure.core/map` and clojure's `loop`/`recur` constructs. Like
`map`, `fork` allows you to map over a seq, generating a single value per element of the input seq. Like
`loop`/`recur`, `fork` allows you to specify initialization values for some locally-bound variables, and then
to update the values of those variables on a series of iterations. Let us consider the mapping functionality
in isolation before discussing in more depth the recurrence functionality:

```
(r/evaluate
  (G (fork [:a (slices [1 2 3])]
           [] ; This argument is used for the `loop/recur`-like functionality.
           (* :a 2)
           [] ; Also used for `loop/recur`-like functionality.
           )))
;; => #Tensure [[    2.0000,    4.0000,    6.0000]]
;; The result is a 3-element vector.
```

We will discuss the second and last arguments to `fork` shortly, but for now consider the first and third
arguments. The first is a set of "input bindings": just as clojure's `let` allows you to bind symbols to
values locally (in the body of the `let`), `fork` allows you to bind input nodes to values of some seq-valued
node. The third argument (the 'output node') is evaluated one time for each value in the input seq, and the
overall result of the `fork` is the concatenation of those values. The analogs operation in plain clojure
would be `(map (fn [a] (* a 2)) [1 2 3])` or

```
(for [a [1 2 3]]
  (* a 2))
```

`fork` is therefore often used in conjunction with ops that produce seqs of tensors: `slices` and
`partition-along`.

```
; Get a seq of slices along the 0-th axis. Each element of the seq will be a scalar, since the input is
; a vector.
(r/evaluate (G (slices [1 2 3])))
;; => (#Tensure 1.0000
;;     #Tensure 2.0000
;;     #Tensure 3.0000)

; Get a seq of slices along the 1-st axis (i.e. a seq of column vectors for a matrix input).
(def a (r/evaluate (G (slices [[1 2 3] [4 5 6]] 1))))
;; => (#Tensure [[    1.0000,    4.0000]]
;;     #Tensure [[    2.0000,    5.0000]]
;;     #Tensure [[    3.0000,    6.0000]])
(m/shape (first a))
;; => [2]

; Partition along axis 0 into 3-element vectors spaced 2 elements apart.
(def b (r/evaluate (G (partition-along [1 2 3 4 5] 0 3 2))))
;; => (#Tensure [[    1.0000,    2.0000,    3.0000]]
;;     #Tensure [[    3.0000,    4.0000,    5.0000]]
;;     #Tensure 5.0000)
(map m/shape b)
;; => ([3] [3] [1])

; Partition along axis 1 into 2-element matrices spaced 2 elements apart.
(def c (r/evaluate (G (partition-along [[1 2 3 4 5 6] [7 8 9 10 11 12]] 1 2))))
;; => (#Tensure [[    1.0000,    2.0000],
;;               [    7.0000,    8.0000]]
;;     #Tensure [[    3.0000,    4.0000],
;;               [    9.0000,   10.0000]]
;;     #Tensure [[    5.0000,    6.0000],
;;               [   11.0000,   12.0000]])
(map m/shape c)
;; => ([2 2] [2 2] [2 2])
```

Note that `slices` returns a seq of tensors with one less dimension than the input tensor, while
`partition-along` returns a seq of tensors that are the same dimensionality as the input tensor. It is
possible to use tensors of any dimensionality as `fork` inputs, to have multiple inputs, and to have input
seqs provided directly as inputs or constants:

```
(r/evaluate
  (G (fork [:a (slices [1 0 -1])
            :b (partition-along [[1 2 3 4 5 6] [7 8 9 10 11 12]] 1 2)]
           []
           (* :a :b)
           [])))
;; => #Tensure [[[    1.0000,    2.0000],
;;               [    7.0000,    8.0000]],
;;
;;              [[         0,         0],
;;               [         0,         0]],
;;
;;              [[   -5.0000,   -6.0000],
;;               [  -11.0000,  -12.0000]]]
; Note that the output is a single tensor produced by stacking the output values from each iteration
; along the 0-th axis. It's also possible to stack them along a different dimension:

(def b (r/evaluate
         (G (fork [:a (slices [1 0 -1])
                   :b (partition-along [[1 2 3 4 5 6] [7 8 9 10 11 12]] 1 2)]
                  []
                  (* :a :b)
                  []
                  {:output-time-axis 1}))))
;; => #Tensure [[[    1.0000,    2.0000],
;;               [         0,         0],
;;               [   -5.0000,   -6.0000]],
;;
;;              [[    7.0000,    8.0000],
;;               [         0,         0],
;;               [  -11.0000,  -12.0000]]]
(m/shape b)
;; => [2 3 2]
```

Now consider the following example of the recurrence functionality. Note that the input binding is required
for determining the number of iterations but is not used to determine the output:

```
(r/evaluate
  (G (fork [:a (slices [1 2 3 4])]
           [:b 1]
           :b
           [(* :b -2)])))
;; => #Tensure [[    1.0000,   -2.0000,    4.0000,   -8.0000]]

(r/evaluate
  (G (fork [:a (slices [1 2 3 4])]
           [:b 7]
           :b
           [(* :b -2)])))
;; => #Tensure [[    7.0000,  -14.0000,   28.0000,  -56.0000]]

(r/evaluate
  (G (fork [:a (slices [1 2 3 4])]
           [:b 2]
           :b
           [(pow :b 2)])))
;; => #Tensure [[    2.0000,    4.0000,   16.0000,  256.0000]]
```

The second argument to `fork` is a set of bindings for nodes whose values will be updated with each
iteration. These bindings play the same role as the bindings in a Clojure `loop`. The fourth argument is a seq
of nodes that update the values of the recurrent bindings. These play the same role as the expressions in a
Clojure `recur`. Like a clojure `loop/recur`, there can be multiple bindings:

```
(r/evaluate
  (G (fork [:a (slices [1 2 3 4])]
           [:b 1
            :c 10]
           (+ :b :c)
           [(+ :b 1)
            (* :c 2)])))
;; => #Tensure [[   11.0000,   22.0000,   43.0000,   84.0000]]
```

Both sets of bindings used in `fork`s--the input bindings and the recurrent node bindings--occur
progressively, just as do bindings in a clojure `let`--i.e. just as the first symbol in a clojure `let` can be
used in the expression whose value is assigned to the second symbol, the first input node in a Ranvier `fork`
binding can be used in the expression for the second binding, and so on:

```
(r/evaluate
 (G (fork [:a (slices [[1 2 3 4]
                       [5 6 7 8]] 1)]
           [:b (* :a 2)
            :c (* :b 10)]
           (join-along 0 (reshape :b [1 2]) (reshape :c [1 2]))
           [(+ :b 1)
            (* :c 2)])))
;; => #Tensure [[[    2.0000,   10.0000],
;;               [   20.0000,  100.0000]],
;;
;;              [[    3.0000,   11.0000],
;;               [   40.0000,  200.0000]],
;;
;;              [[    4.0000,   12.0000],
;;               [   80.0000,  400.0000]],
;;
;;              [[    5.0000,   13.0000],
;;               [  160.0000,  800.0000]]]
```

Without `fork`, each node in a Ranvier graph would have exactly one value. But `fork` introduces another
dimension to values: the nodes bound in a `fork` have values that change as the `fork` iterates. Yet at the
end of a forward propagation step, each node must still have a single value:

```
(-> (r/forward
      (G (named :f
                (fork [:a (slices [1 2 3 4])]
                      [:b 1]
                      :b
                      [(* :b -2)]))))
    (select-keys [:f :a :b]))
;; => {:f #Tensure [[    1.0000,   -2.0000,    4.0000,   -8.0000]]
;;     :a #Tensure 4.0000
;;     :b #Tensure 16.0000}
```

After one step of forward propagation, `:a` is bound to the same value it had on the last iteration, and `:b`
is similarly bound to the last value it took. Note that the values for recurrent nodes (`:b` in this example)
are evaluated at the end of every iteration, including the last one: in the above example, `:b` is 1 during
the first iteration (when the output node value is computed); at the end of the first iteration, `:b` is
updated to -2. It's update to 4 at the end of the second iteration and to -8 at the end of the third. It is
therefore -8 during the fourth iteration, but at the end of the fourth iteration, it is updated to 16, even
though there are no subsequent iterations.

Within `fork`s there is a concept of "time" (time advances with each iteration). But outside of `fork`s, no
such concept exists: each node has only one value. If using Ranvier for modeling neural networks, it is
helpful to think of propagation as occurring in the following pattern: 1) values propagate to all `fork` nodes,
2) the recurrent subnetworks in `fork`s propagate the values until they reach 'stable' states (i.e. all
iterations within the `fork`s are complete), 3) values propagate out of `fork`s into downstream parts of the
network. However, since locally bound inputs in `fork`s have multiple different values during a single run of
the graph, allowing their use outside of the `fork` would introduce ambiguity: which values of these nodes
should be used in values outside the `fork`, where there is no concept of time or iteration? It is therefore
not possible to use these values outside of a `fork`:

```
(r/evaluate
  (G (* :a
        (fork [:a (slices [1 2 3 4])]
              [:b 1]
              :b
              [(* :b -2)]))))
;; => Unhandled java.lang.Exception
;;    Invalid graph. Node ':a' is used out of scope.

(r/evaluate
  (G (* :b
        (fork [:a (slices [1 2 3 4])]
              [:b 1]
              :b
              [(* :b -2)]))))
;; => Unhandled java.lang.Exception
;;    Invalid graph. Node ':b' is used out of scope.
```

Even though it is not possible to use locally bound inputs outside of the `fork` where they're defined, they
still have unique values at the end of a run of forward propagation, and a single gradient can be given for
them at the end of back propagation. In order for this to be the case, two different locally bound inputs
cannot share the same name, even if their scope is restricted:

```
(r/evaluate
  (G (+ (fork [:a (slices [1 2 3 4])]
              [:b 1]
              :b
              [(* :b -2)])
        (fork [:a (slices [1 2 3 4])]
              [:b 1]
              :b
              [(* :b -2)]))))
;; => Unhandled java.lang.Exception
;;    Invalid graph. Node ':a' is used out of scope.

(r/evaluate
  (G (+ (fork [:a (slices [1 2 3 4])]
              [:b 1]
              :b
              [(* :b -2)])
        (fork [:c (slices [1 2 3 4])]
              [:d 1]
              :d
              [(* :d -2)]))))
;; => #Tensure [[    2.0000,   -4.0000,    8.0000,  -16.0000]]

(r/evaluate
  (G (let [fork-subnetwork (fork [:a (slices [1 2 3 4])]
                                 [:b 1]
                                 :b
                                 [(* :b -2)])]
       (+ fork-subnetwork fork-subnetwork))))
;; => #Tensure [[    2.0000,   -4.0000,    8.0000,  -16.0000]]
```

In the first case above, two identical `fork` nodes are constructed, and we get an Exception because both
`fork`s use the same local input names, and this is not allowed. The second case works, because we use
different sets of local input names in the two `fork`s. The third case works because the graph includes only a
single `fork` node, though it has two "connections". The fact that the third case works and the first case
does not, even though the two cases appear to be superficially equivalent, is important: _unless nodes are
identified by name, each call to an op generates a new node_.

### Custom ops

Ranvier provides a macro, `defop`, for defining custom ops:

```
(require '[ranvier.core :refer [G defop]])

(defop times-three   ; <- `defop` followed by op name
  [^:nd n]   ; <- vector of bindings for the op's operands. ^:nd is a type hint to tell Ranvier that `n` is
             ;    an n-dimensional (nd) array (tensor).
  :v (m/mul n (m/array 3))   ; <- we define the computation and give the output value a name (`:v`)
  :dn (m/mul dv (m/array 3)))   ; <- if the output of the full computation graph is `y`, then `dy/dv` is
                                     available here as `dv`: the gradient of graph output with respect to
                                     the output of this node (named `:v`). In order to backpropagate through
                                     this node, we must specify how to compute `dy/dn`, which is done here.
(r/evaluate (G (times-three 7)))
;; => #Tensure 21.0000

(let [g (G (times-three :a))
      vals (r/forward g {:a 7})]
  (r/backward g vals [:a]))
;; => {:a #Tensure 3.0000}
```

Defining a custom op in Ranvier is as simple as defining: 1) the forward computation, and 2) the gradients of
each operand in terms of the op node's gradient and the operand values. Although most of Ranvier's ops are
defined to work exclusively on Tensure tensors, as an example, we can define ops that work on regular clojure
numbers:

```
(defop a-times-b
  [a b]
  :out (* a b)
  :da (* b dout)
  :db (* a dout))

(r/evaluate (a-times-b 3 7))
; => 21

(let [g (G (a-times-b :a :b))
      vals (r/forward g {:a 3 :b 7})]
  (r/backward g vals [:a :b] 1))
; => {:b 3, :a 7}
```

Note that `a-times-b` evaluates to a `java.lang.Long` rather than to a Tensure object. The gradients are also
`Long`s rather than `Tensure`s. Note that we must provide a starting gradient (i.e. the gradient of the final
graph output node) to `backward` as the fourth argument, because if we did not Ranvier would supply a gradient
of `#Tensure 1.0` (i.e. a `Tensure` scalar with value `1.0`), which is incompatible with our definition of
the gradients. (Optimization also will not work, because Ranvier assumes that all parameters to be optimized
are Tensures. This is just a toy example.)

We can make also make version that takes an arbitrary number of operands:

```
(defop identity-op
  [a]
  :v a
  :da dv)

(defn times
  [& operands]
  (if (= (count operands) 1)
    (identity-op (first operands))
    (reduce a-times-b operands)))

(let [g (G (named :out (times :b (times :a))))
      vals (r/forward g {:a 3 :b 7})]
  (println "Result: " (:out vals))
  (println "Gradients: " (r/backward g vals [:a :b] 1)))
;; Prints:
;;   Result:  21
;;   Gradients:  {:b 3, :a 7}

(let [g (G (named :out (times :a :b :c)))
      vals (r/forward g {:a 3 :b 7 :c 11})]
  (println "Results: " (:out vals))
  (println "Gradients: " (r/backward g vals [:a :b :c] 1)))
;; Prints:
;;   Results:  231
;;   Gradients:  {:c 21, :b 33, :a 77}

```

## Constructing graphs

### Node types

As implied above, Ranvier has three types of nodes:

  - _constants_ - these have values that are defined at the time of graph construction; their values cannot
    change
  - _inputs_ - these have values that must be provided at graph runtime (or, in the case of locally bound
    inputs in `fork`, that must be set from the value of another node at graph runtime)
  - _ops_ - these are nodes constructed by ops - their values are calculated during forward propagation, not
    provided directly

_op_ nodes are constructed every time you use an op. _constant_ nodes are usually constructed by embedding a
constant directly in the graph definition. _input_ nodes are typically defined using a keyword. `(G (+ 1
:a))`, for example, constructs three nodes: a constant node (with value `1`), an input node named `:a`, and an
`op` node whose value will be determined during forward propagation to equal (+ 1 :a). Although constant and
input nodes will typically be constructed "implicitly", it is also possible to construct them explicitly using
the functions `const` and `input`. For example `(G (+ (r/const (m/array 1)) (r/input :a)))` is equivalent to
`(G (+ 1 :a))`. Note, however, that in this case the Tensure object for `1` must be constructed explicitly as
well.

```
(r/evaluate (G (+ 1
                  :a))
            {:a 3})
;; => #Tensure 4.0000

(r/evaluate (G (+ (r/const (m/array 1))
                  (r/input :a)))
            {:a 3})
;; => #Tensure 4.0000
```

Explicit construction is useful primarily for providing additional information about the node at
construction-time. The section on [initialization](#initialization) provides some examples of this type of
scenario.

### Type inference

Nodes have types (_constant_, _input_, or _op_), and their values also have types: most node values are
Tensures, but some are other types (e.g. a tensor shape represented as a clojure vector, a map of options, or
a keyword). For convenience, Ranvier allows you to provide tensors as Clojure vectors or numbers, and it will
automatically construct Tensure objects from them. But it will only do this if it can infer that the constant
or input should be a tensor (and should not be left as a regular Clojure vector).  Ranvier's type inference
works by type hints provided in op definitions: if an op provides an `^:nd` type hint indicating that the
operand must be a tensor, then any number of `PersistentVector` arguments provided for that operand are
automatically converted into constant nodes.

```
;; The following two ops are identical, except the latter has an `^:nd` type hint.
(defop any-identity
  [a]
  :v a
  :da dv)

(defop tensor-identity
  [^:nd a]
  :v a
  :da dv)

(r/evaluate (any-identity [1 2 3]))
;; => [1 2 3]

(r/evaluate (tensor-identity [1 2 3]))
;; => #Tensure [[    1.0000,    2.0000,    3.0000]]

(r/evaluate (any-identity 1))
;; => 1

(r/evaluate (tensor-identity 1))
;; => #Tensure 1.0000

; In the above examples, `any-identity` leaves `[1 2 3]` and `1` as-is, while `tensor-identity` converts
; them into Tensure objects.

; Type inference and automatic conversion also applies to inputs:
(r/evaluate (any-identity :a) {:a [1 2 3]})
;; => [1 2 3]

(r/evaluate (tensor-identity :a) {:a [1 2 3]})
;; => #Tensure [[    1.0000,    2.0000,    3.0000]]
```

### Initialization

There are some cases where it's useful to assign a node a value the first time a graph is run based on the
values of other nodes. For example, suppose we want to calculate `(mmul :some-parameter :some-input)`, and we
don't know the value of `:some-input` in advance. We want to optimize `:some-parameter`, starting from some
initial random value. In this case, we can explicitly construct an input node for `:some-parameter` and
provide it with an initialization function. At runtime, this function will receive a map of node values
computed thus far. It must return the starting value for the node.

```
(require '[tensure.core :as m])
(def g (G (mmul (r/input :some-parameter
                         (fn [input-map]
                           (m/sample-uniform [1 (m/row-count (:some-input input-map))])))
                :some-input)))

(r/forward g {:some-input [[1 2]
                           [3 4]]})
;; => {:some-input #Tensure [[    1.0000,    2.0000],
;;                           [    3.0000,    4.0000]]
;;     :some-parameter #Tensure [[    0.1434,    0.2846]]
;;     :mmul34601 #Tensure [[    0.9970,    1.4250]]}

(r/forward g {:some-input [[1 2]
                           [3 4]
                           [5 6]]})
;; => {:some-input #Tensure [[    1.0000,    2.0000],
;;                           [    3.0000,    4.0000],
;;                           [    5.0000,    6.0000]]
;;     :some-parameter #Tensure [[    0.8396,    0.1535,    0.3660]]
;;     :mmul34601 #Tensure [[    3.1299,    4.4889]]}
```

In the above example, `:some-parameter` will always be initialized to a row vector that has the same number of
columns as there are rows in `:some-input`.

The only guarantee associated with initialization is that _for any given single op node, operands with
initializers will be evaluated after operands without initializers_. If a single op node has multiple operands
with initializers, the order in which they will be initialized is not guaranteed. The presence of values in
the `input-map` for nodes that are not siblings of the node being initialized is also not guaranteed.

## Optimization

Ranvier provides a pluggable framework for optimization that includes a number of different functions. Since
the details of this framework are a work-in-progress, this section provides just a brief summary of a single
function, `simple-optimize`, which will be sufficient for many purposes.

### Optimization results

`simple-optimize` takes two arguments: a graph and a map of options. If the option `:full-result` is set to
`true`, then `simple-optimize` returns a full optimization result object; if it's `false` (the default), it
returns only a map of optimized parameter values. When provided as the `:prev-result` option, the full
optimization result includes all the information necessary for `simple-optimize` to resume optimization from
the point at which it left off:

```
(def graph (G (-> (pow (- :a 3) 2)
                  (report (r/make-value-reporter "Value") (r/make-print-logger :space)))))

; When `:full-result` is `false`, `simple-optimize` returns the parameter values.
(r/simple-optimize graph {:iterations 100
                          :hyperparams {:learning-rate 0.1}})
;; => {:a #Tensure 3.0000}

(def prev-result (r/simple-optimize graph {:iterations 2
                                           :hyperparams {:learning-rate 0.1
                                                         :report [:iteration]}
                                           :full-result true}))
;; Print:
;;   Value: 9.0  Iteration: 1
;;   Value: 5.76  Iteration: 2

(def next-result (r/simple-optimize graph {:iterations 5
                                           :hyperparams {:learning-rate 0.1
                                                         :report [:iteration]}
                                           :full-result true
                                           :prev-result prev-result}))
;; Prints:
;;   Value: 3.6864  Iteration: 3
;;   Value: 2.359296  Iteration: 4
;;   Value: 1.5099496  Iteration: 5
;;   Value: 0.96636784  Iteration: 6
;;   Value: 0.6184753  Iteration: 7

(r/simple-optimize graph {:iterations 3
                          :hyperparams {:learning-rate 0.1
                                        :report [:iteration]}
                          :prev-result next-result})
;; Prints:
;;   Value: 0.39582422  Iteration: 8
;;   Value: 0.2533274  Iteration: 9
;;   Value: 0.16212961  Iteration: 10

```

### Optimizer

Ranvier can make use of arbitrary optimization functions that support a defined interface. It also provides a
gradient descent-based optimizer that will suffice for many cases, including machine learning applications
that require optimization over difficult, non-convex terrains. The `:optimizer` option may be either a
function or keyword for useful presets: `:sgd` (stochastic gradient descent), `:batch-gd` (batch gradient
descent), `:momentum` (gradient descent with momentum), `:rms-prop` (RMSProp), and `:adam` (a version of the
Adam algorithm that is just RMSProp + momentum).

The `:hyperparams` option can be a map that is passed directly to the optimizer. The [API
docs](https://cguenthner.github.io/ranvier/docs/index.html) include details on the options supported by
`gradient-descent-optimizer`. The above example showed how to set `gradient-descent-optimizer`s learning rate,
as well as the `:report` option. `:report` is a seq of keys for information that should be logged on each
iteration. Possible keys include: `:iteration`, `:iteration-duration`, `:epoch`, `:epoch-duration`, `:batch`,
`:batch-count`, `:elapsed-time`, `:mean-iteration-duration`, `:epoch-time-remaining`, and
`:time-remaining`. Note that this logging occurs at the _end_ of each iteration, so it will follow any logging
from the `report` op.

### Batching

Batch optimization is enabled by specifying `:batch-inputs` and/or `:batch-size` hyperparams. The former is a
map of input names to axes that should be split to generate the batches, and the latter is the target size of
each subtensor along the split dimension. If `:batch-inputs` is omitted but `:batch-size` is provided, then
_all_ inputs will be batched along the 0th dimension. The last batch may be smaller than `:batch-size`.

```
(def g (G (report :a
                  (r/make-value-reporter ":a")
                  (r/make-print-logger :space))))
(def inputs {:a (m/array [[1 2 3 4 5 6]
                          [7 8 9 10 11 12]
                          [13 14 15 16 17 18]
                          [19 20 21 22 23 24]
                          [25 26 27 28 29 30]])})

; (Note that there are no parameters, so nothing is being optimized here. We're just running the graph
; for a defined number of iterations.)
; No batching.
(r/simple-optimize g {:hyperparams {:report [:iteration :batch :epoch]
                                    :iterations 2}
                      :inputs inputs})
;; Prints:
;;   :a: #Tensure [[  1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000],
;;                 [  7.0000,  8.0000,  9.0000, 10.0000, 11.0000, 12.0000],
;;                 [ 13.0000, 14.0000, 15.0000, 16.0000, 17.0000, 18.0000],
;;                 [ 19.0000, 20.0000, 21.0000, 22.0000, 23.0000, 24.0000],
;;                 [ 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000]]  Iteration: 1  Batch: 1  Epoch: 1
;;   :a: #Tensure [[  1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000],
;;                 [  7.0000,  8.0000,  9.0000, 10.0000, 11.0000, 12.0000],
;;                 [ 13.0000, 14.0000, 15.0000, 16.0000, 17.0000, 18.0000],
;;                 [ 19.0000, 20.0000, 21.0000, 22.0000, 23.0000, 24.0000],
;;                 [ 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000]]  Iteration: 2  Batch: 1  Epoch: 2

; Split all inputs into batches of size 2 along axis 0
(r/simple-optimize g {:hyperparams {:report [:iteration :batch :epoch]
                                    :batch-size 2
                                    :iterations 6}
                      :inputs inputs})
;; Prints:
;;   :a: #Tensure [[  1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000],
;;                 [  7.0000,  8.0000,  9.0000, 10.0000, 11.0000, 12.0000]]  Iteration: 1  Batch: 1  Epoch 1
;;   :a: #Tensure [[ 13.0000, 14.0000, 15.0000, 16.0000, 17.0000, 18.0000],
;;                 [ 19.0000, 20.0000, 21.0000, 22.0000, 23.0000, 24.0000]]  Iteration: 2  Batch: 2  Epoch 1
;;   :a: #Tensure [[ 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000]]  Iteration: 3  Batch: 3  Epoch 1
;;   :a: #Tensure [[  1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000],
;;                 [  7.0000,  8.0000,  9.0000, 10.0000, 11.0000, 12.0000]]  Iteration: 4  Batch: 1  Epoch 2
;;   :a: #Tensure [[ 13.0000, 14.0000, 15.0000, 16.0000, 17.0000, 18.0000],
;;                 [ 19.0000, 20.0000, 21.0000, 22.0000, 23.0000, 24.0000]]  Iteration: 5  Batch: 2  Epoch 2
;;   :a: #Tensure [[ 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000]]  Iteration: 6  Batch: 3  Epoch 2
; Note that the last input is of size 1, because the total number of rows is not evenly divisible by the
; batch size.

; Split input :a into batches of size 3 along axis 1.
(r/simple-optimize g {:hyperparams {:report [:iteration :batch :epoch]
                                    :batch-size 3
                                    :batch-inputs {:a 1}
                                    :iterations 4}
                      :inputs inputs})
;; Prints:
;;   :a: #Tensure [[    1.0000,    2.0000,    3.0000],
;;                 [    7.0000,    8.0000,    9.0000],
;;                 [   13.0000,   14.0000,   15.0000],
;;                 [   19.0000,   20.0000,   21.0000],
;;                 [   25.0000,   26.0000,   27.0000]]  Iteration: 1  Batch: 1  Epoch: 1
;;   :a: #Tensure [[    4.0000,    5.0000,    6.0000],
;;                 [   10.0000,   11.0000,   12.0000],
;;                 [   16.0000,   17.0000,   18.0000],
;;                 [   22.0000,   23.0000,   24.0000],
;;                 [   28.0000,   29.0000,   30.0000]]  Iteration: 2  Batch: 2  Epoch: 1
;;   :a: #Tensure [[    1.0000,    2.0000,    3.0000],
;;                 [    7.0000,    8.0000,    9.0000],
;;                 [   13.0000,   14.0000,   15.0000],
;;                 [   19.0000,   20.0000,   21.0000],
;;                 [   25.0000,   26.0000,   27.0000]]  Iteration: 3  Batch: 1  Epoch: 2
;;   :a: #Tensure [[    4.0000,    5.0000,    6.0000],
;;                 [   10.0000,   11.0000,   12.0000],
;;                 [   16.0000,   17.0000,   18.0000],
;;                 [   22.0000,   23.0000,   24.0000],
;;                 [   28.0000,   29.0000,   30.0000]]  Iteration: 4  Batch: 2  Epoch: 2
```

### Async optimization

When the `:async` option is set to `true`, `simple-optimize` will launch optimization in a new thread and
return immediately rather than blocking until optimization is complete (the default behavior). The return
value is a function that when called will stop optimization and return either the full result or a map of
optimized parameter values.

```
; This op just propagates values forward and backward unchanged, but introduces a delay for demonstration
; purposes.
(defop slow-it-down
  [a]
  :v (do (Thread/sleep 100)
         a)
  :da dv)

(def graph (G (slow-it-down (pow (- :a 3) 2))))
(r/simple-optimize graph {:iterations 100
                          :hyperparams {:learning-rate 0.1
                                        :report [:iteration :mean-iteration-duration]}})
; About every 100 ms, 'Iteration # Mean iteration duration: 1xx ms' is printed
; After ~10 s (100 iterations * 100 ms/iteration):
;; => {:a #Tensure 3.0000}

(def stop-optimization (r/simple-optimize graph {:full-result true
                                                 :hyperparams {:learning-rate 0.1
                                                               :report [:iteration :mean-iteration-duration]}
                                                 :async true}))
; `simple-optimize` immediately returns:
;; => #function[ranvier.core/simple-optimize/fn--10922]
; Every 100 ms, 'Iteration # Mean iteration duration: 1xx ms' is printed

(Thread/sleep 5000)
(def result (stop-optimization))
;; => an optimization result, returned after the next iteration is complete
; Optimization stops at ~iteration 50

; We can resume optimization from the point at which we stopped.
(def get-result (r/simple-optimize graph {:prev-result result
                                          :hyperparams {:learning-rate 0.1
                                                        :report [:iteration :mean-iteration-duration]}
                                          :async true}))
;; => #function[ranvier.core/simple-optimize/fn--10922]
; About every 100 ms, 'Iteration # Mean iteration duration: 1xx ms' is printed, starting at ~iteration 50

(Thread/sleep 5000) ; Allow enough time for ~50 more iterations.
(get-result)
;; => {:a #Tensure 3.0000}
```

Note that with asynchronous optimization, `:iterations` is not specified. Optimization will continue
indefinitely until the `stop-optimization` function is called.

### Inputs, parameters, and state

As described in the section on [node types](#node-types), input nodes have values that must be provided before
the graph can be run (or else an initializer must be provided). In addition, input node values can be
calculated from a subgraph within a `fork`. Thus, input nodes are used in a few different ways in Ranvier:

  - as true _inputs_ - these have values that are provided and that do not change within the context of a
    particular optimization problem. Example: the training data in a machine learning problem.
  - as _parameters_ - these have values that the optimizer will change after each iteration of optimization;
    their initial values must be provided (or else an initializer must be provided), but otherwise their values
    are varied to minimize the value of the graph's root node. Example: the weight matrix in a densely
    connected forward neural network model
  - as _state_ - these have values that will change as a result of merely running the graph; their values may
    change many times within a single iteration. Example: a recurrence target in `fork`; the 'state' value of
    a recurrent neural network layer.

You can provide values for each of these types of inputs using the `:inputs`, `:params`, and `:state` options
to `simple-optimize`.

```
(def g (G (-> (fork [:a (->> (range 0 5)
                             (map m/array))]
                    [:counter :counter]
                    :counter
                    [(+ :counter :step-size)])
              (report (r/make-value-reporter "Output") (r/make-print-logger :newline)))))

(def result (r/simple-optimize g {:iterations 3
                                  :inputs {:step-size 2}
                                  :state {:counter 0}
                                  :full-result true}))
;; Prints:
;;   Output: #Tensure [[         0,    2.0000,    4.0000,    6.0000,    8.0000]]
;;   Output: #Tensure [[   10.0000,   12.0000,   14.0000,   16.0000,   18.0000]]
;;   Output: #Tensure [[   20.0000,   22.0000,   24.0000,   26.0000,   28.0000]]

; State is maintained in the optimization result, so we can resume optimization where we left off.
(r/simple-optimize g {:iterations 3
                      :prev-result result})
;; Prints:
;;   Output: #Tensure [[   30.0000,   32.0000,   34.0000,   36.0000,   38.0000]]
;;   Output: #Tensure [[   40.0000,   42.0000,   44.0000,   46.0000,   48.0000]]
;;   Output: #Tensure [[   50.0000,   52.0000,   54.0000,   56.0000,   58.0000]]

; We can change `:step-size` and the starting value of `:counter`.
(r/simple-optimize g {:iterations 3
                      :inputs {:step-size -3}
                      :state {:counter 100}})
;; Prints:
;;   Output: #Tensure [[  100.0000,   97.0000,   94.0000,   91.0000,   88.0000]]
;;   Output: #Tensure [[   85.0000,   82.0000,   79.0000,   76.0000,   73.0000]]
;;   Output: #Tensure [[   70.0000,   67.0000,   64.0000,   61.0000,   58.0000]]
```

The `fork` in the above example assigns the initial value of the local `:counter` input to be the value of
`:counter` itself, which on the first iteration is read from the `:state` map provided to
`simple-optimize`. The `fork` takes 5 steps on each iteration, changing `:counter` with each step. As a
_state_ input, the value of `:counter` is maintained across optimization runs. This type of recurrence--where
a local input's initial value in a fork depends on its own value--is the only way in which state can enter a
Ranvier graph.

## Contributing

If you find a bug or would like a feature, then please open a GitHub issue. Even better, please submit a pull
request.

## License

Copyright © 2019 Casey Guenthner

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
