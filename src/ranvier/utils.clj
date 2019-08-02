(ns ranvier.utils
  (:require [clojure.set :as set]))

(defmacro def-
  "Like `def` but defines the var to be local to the current namespace."
  [sym & init]
  `(def ~(with-meta sym {:private true}) ~@init))

(defmacro -?> [& args] `(some-> ~@args))
(defmacro -?>> [& args] `(some->> ~@args))

(defn throw-str
  "Throws an `Exception` with a message produced by `str`ing together `args`."
  [& args]
  (let [; Printing giant Exception messages freezes Cider, so we truncate them after a certain length.
        max-throw-str-len 10000
        ; Ellipsis is three characters.
        max-msg-len (- max-throw-str-len 3)
        s (apply str args)
        message (if (<= (count s) max-msg-len)
                  s
                  (str (subs s 0 max-msg-len) "..."))]
    (throw (Exception. message))))

(defn update-vals
  "Given a map `m`, a function `f`, and an arbitrary number of aditional arguments, returns a map with the
  same keys as `m` with the values updated to the value of `(f old-value args)."
  [m f & args]
  (->> (for [[k v] m] [k (apply f v args)])
       (into {})))

(defn zip
  "Zips two collections together. E.g. (zip [1 2 3] [4 5 6]) returns ([1 4] [2 5] [3 6])."
  [& colls]
  (if (zero? (count colls))
    []
    (apply map vector colls)))

; TODO: Unit test
(defn unzip
  [c]
  (reduce (fn [unzipped el]
            (map conj unzipped el))
          (repeat (count (first c)) [])
          c))

; Modified from https://stackoverflow.com/questions/14836414/can-i-make-a-deterministic-shuffle-in-clojure
(defn deterministic-shuffle
  ([^java.util.Collection coll]
   (deterministic-shuffle 0 coll))
  ([seed ^java.util.Collection coll]
   (let [al (java.util.ArrayList. coll)
         rng (java.util.Random. seed)]
     (java.util.Collections/shuffle al rng)
     (clojure.lang.RT/vector (.toArray al)))))

(defn check-key-validity
  "Throws an error if map `m` does not have one of the `required-keys` and has any invalid keys (not in
  `required-keys` or `optional-keys`. Returns `nil`."
  [m required-keys optional-keys]
  (when-not (map? m)
    (throw-str "Cannot check options map. Received options was of type '" (type m) "', not a map."))
  (let [required-key-set (into #{} required-keys)
        allowed-key-set (into required-key-set optional-keys)
        present-key-set (into #{} (keys m))
        missing-keys (vec (set/difference required-key-set present-key-set))
        invalid-keys (vec (set/difference present-key-set allowed-key-set))]
    (when (seq missing-keys)
      (throw-str "Missing required keys " missing-keys " in: " m "."))
    (when (seq invalid-keys)
      (throw-str "Invalid keys " invalid-keys " in: " m "."))))

(def profile-timings (atom {}))
(defmacro with-profiling
  "Accumulates time (in ms) spent executing `body`."
  [key & body]
  `(let [start# (System/nanoTime)
         result# (do ~@body)
         elapsed-ms# (double (/
                               (- (System/nanoTime) start#)
                               1e6))]
     (swap! profile-timings update ~key #(+ (or %1 0) (or %2 0)) elapsed-ms#)
     result#))

(defn reset-profiling!
  []
  (reset! profile-timings {}))

(defn print-profile
  []
  (doseq [[key time-ms] @profile-timings]
    (println (format "%s: %.2f ms" key time-ms))))

; TODO: Unit test
(defn cartesian-product
  [s & seqs]
  (if (seq seqs)
    (let [seqs-cartesian-product (apply cartesian-product seqs)]
      (mapcat #(map (partial cons %) seqs-cartesian-product) s))
    (map vector s)))

; From https://stackoverflow.com/questions/3407876/how-do-i-avoid-clojures-chunking-behavior-for-lazy-seqs-that-i-want-to-short-ci
(defn unchunk
  "Returns a lazy-seq that does not use chunking."
  [s]
  (when (seq s)
    (lazy-seq
      (cons (first s)
            (unchunk (next s))))))

(defn rand-int-between
  "Returns a random integer in [lower, upper)."
  [lower upper]
  (+ lower (rand-int (- upper lower))))

(defn find-first
  "Returns the first element e in coll for which (pred e) returns a truthy value."
  [pred coll]
  (first (filter pred coll)))

(defn indistinct
  "Returns a seq of distinct values that are present more than once in seq `s`, or `nil` if all values in
  `s` are distinct."
  [s]
  (->> s
       (reduce (fn [[indistinct seen] el]
                 [(if (seen el)
                    (conj indistinct el)
                    indistinct)
                  (conj seen el)])
               [[] #{}])
       first
       seq))
