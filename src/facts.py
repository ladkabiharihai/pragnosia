"""The factual probe set, as DATA ONLY.

This lived inside exp/factcheck_ladder.py, and importing it from another script re-executed that whole file:
it loaded four checkpoints and started running the benchmark as a side effect of `from factcheck_ladder import
FACTS`. This project has hit that exact trap before, when `from muon import Muon` re-ran a 24.5M experiment and
inherited sys.argv. A module that is imported for its data must contain no top-level work.

Each entry is (stem, correct completion, three distractors of the SAME KIND). Distractors share the answer's
category on purpose: a model can pick the right option for the wrong reason when the wrong options are not
even the right kind of thing.
"""
# stem, correct, three distractors of the SAME KIND
FACTS = [
 ("The capital of France is"," Paris",[" Berlin"," Madrid"," Rome"]),
 ("The capital of Japan is"," Tokyo",[" Beijing"," Seoul"," Bangkok"]),
 ("The capital of Italy is"," Rome",[" Milan"," Naples"," Turin"]),
 ("The capital of Egypt is"," Cairo",[" Luxor"," Aswan"," Giza"]),
 ("The capital of Canada is"," Ottawa",[" Toronto"," Montreal"," Vancouver"]),
 ("The capital of Russia is"," Moscow",[" Kiev"," Minsk"," Warsaw"]),
 ("The largest planet in our solar system is"," Jupiter",[" Saturn"," Neptune"," Mars"]),
 ("The planet closest to the Sun is"," Mercury",[" Venus"," Earth"," Mars"]),
 ("The chemical symbol for gold is"," Au",[" Ag"," Fe"," Cu"]),
 ("The chemical symbol for iron is"," Fe",[" Ir"," In"," Fr"]),
 ("Water is composed of hydrogen and"," oxygen",[" nitrogen"," carbon"," helium"]),
 ("The gas that plants absorb from the air is carbon"," dioxide",[" monoxide"," disulfide"," tetrachloride"]),
 ("The human heart has how many chambers? It has"," four",[" two"," three"," five"]),
 ("The freezing point of water in Celsius is"," zero",[" ten"," thirty"," one hundred"]),
 ("The boiling point of water in Celsius is"," 100",[" 0"," 50"," 200"]),
 ("The speed of light is approximately 300,000 kilometres per"," second",[" hour"," minute"," day"]),
 ("The largest ocean on Earth is the"," Pacific",[" Atlantic"," Indian"," Arctic"]),
 ("The longest river in the world is the"," Nile",[" Amazon"," Danube"," Volga"]),
 ("The tallest mountain in the world is Mount"," Everest",[" Kilimanjaro"," Fuji"," Rainier"]),
 ("The currency of the United States is the"," dollar",[" euro"," pound"," yen"]),
 ("The currency of Japan is the"," yen",[" won"," yuan"," dollar"]),
 ("There are how many days in a week? There are"," seven",[" five"," six"," ten"]),
 ("There are how many months in a year? There are"," twelve",[" ten"," eleven"," twenty"]),
 ("The number of continents on Earth is"," seven",[" five"," six"," eight"]),
 ("The author of Romeo and Juliet is William"," Shakespeare",[" Wordsworth"," Blake"," Faulkner"]),
 ("The theory of relativity was developed by Albert"," Einstein",[" Newton"," Bohr"," Planck"]),
 ("The first president of the United States was George"," Washington",[" Jefferson"," Adams"," Madison"]),
 ("The language spoken in Brazil is"," Portuguese",[" Spanish"," French"," Italian"]),
 ("The language spoken in Austria is"," German",[" Dutch"," Danish"," Polish"]),
 ("A triangle has how many sides? It has"," three",[" four"," five"," six"]),
 ("Two plus two equals"," four",[" three"," five"," six"]),
 ("Ten times ten equals"," one hundred",[" one thousand"," ten"," fifty"]),
 ("The colour of the sky on a clear day is"," blue",[" green"," purple"," orange"]),
 ("Grass is typically the colour"," green",[" blue"," red"," yellow"]),
 ("Bees produce"," honey",[" milk"," silk"," wax paper"]),
 ("Cows produce"," milk",[" honey"," silk"," venom"]),
 ("The animal known as the king of the jungle is the"," lion",[" tiger"," wolf"," bear"]),
 ("Penguins are birds that cannot"," fly",[" swim"," walk"," dive"]),
 ("The Great Wall is located in"," China",[" Japan"," India"," Korea"]),
 ("The Eiffel Tower is located in"," Paris",[" London"," Berlin"," Vienna"]),
 ("The Statue of Liberty is located in New"," York",[" Jersey"," Orleans"," Mexico"]),
 ("The sun rises in the"," east",[" west"," north"," south"]),
 ("Ice is water in its"," solid",[" liquid"," gaseous"," plasma"]),
 ("The organ that pumps blood is the"," heart",[" liver"," kidney"," lung"]),
 ("The organ used for breathing is the"," lungs",[" stomach"," spleen"," pancreas"]),
 ("Doctors who treat teeth are called"," dentists",[" surgeons"," nurses"," chemists"]),
 ("The season that follows winter is"," spring",[" summer"," autumn"," monsoon"]),
 ("A baby dog is called a"," puppy",[" kitten"," calf"," foal"]),
]


# ---------------------------------------------------------------- arithmetic, generated
# Four-way like FACTS, but the distractors are the WRONG ANSWERS A MODEL ACTUALLY GIVES, not random numbers:
# off by one, the carry dropped, the digits concatenated. A model that has only learned the shape of an answer
# ("a two-digit number follows =") scores at chance here, which is the point; random distractors would let it
# score well by format alone. Deterministic in the seed so two checkpoints are compared on the same items.
def arithmetic_items(n=60, seed=0):
    import random
    r = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:                                              # two-digit addition, with and without carry
            a, b = r.randint(10, 99), r.randint(10, 99)
            c = a + b
            wrong = [c - 1, c + 10 if (a % 10 + b % 10) >= 10 else c - 10, int(f"{a}{b}") % 1000]
            stem = f"{a} + {b} ="
        elif kind == 1:                                            # subtraction, never negative
            a, b = r.randint(20, 99), r.randint(1, 19)
            c = a - b
            wrong = [c + 1, c - 10, a + b]
            stem = f"{a} - {b} ="
        elif kind == 2:                                            # single-digit multiplication
            a, b = r.randint(2, 9), r.randint(2, 9)
            c = a * b
            wrong = [c + a, c - b, a + b]
            stem = f"{a} * {b} ="
        else:                                                      # add a small number to a round one
            a, b = r.choice([100, 200, 300, 400, 500]), r.randint(2, 49)
            c = a + b
            wrong = [c - 1, c + 100, a - b]
            stem = f"{a} + {b} ="
        w = []
        for x in wrong:                                            # distinct, non-negative, never the answer
            x = abs(int(x))
            while x == c or x in w:
                x += 1
            w.append(x)
        out.append((stem, f" {c}", [f" {x}" for x in w]))
    return out


# ---------------------------------------------------------------- reasoning, hand-written
# Seven kinds, all four-way with same-kind distractors. "induction" is the one that speaks to this project's
# thesis directly: continuing a pattern seen earlier in the same prompt is exactly what an associative memory
# is for, and a model with no memory of the prompt's start cannot do it.
def reasoning_items():
    return [
 # comparison
 ("Which is larger, 17 or 8? The larger is"," 17",[" 8"," 9"," 25"]),
 ("Which is larger, 4 or 31? The larger is"," 31",[" 4"," 27"," 35"]),
 ("Which is smaller, 62 or 46? The smaller is"," 46",[" 62"," 16"," 108"]),
 ("Which is smaller, 9 or 90? The smaller is"," 9",[" 90"," 81"," 99"]),
 # ordering
 ("Sorted from smallest to largest: 5, 2, 9. The first is"," 2",[" 5"," 9"," 16"]),
 ("Sorted from smallest to largest: 8, 3, 6. The last is"," 8",[" 3"," 6"," 17"]),
 ("Sorted from largest to smallest: 4, 7, 1. The first is"," 7",[" 4"," 1"," 12"]),
 ("In the list 12, 30, 21 the largest is"," 30",[" 12"," 21"," 63"]),
 # transitive
 ("Anna is taller than Ben. Ben is taller than Carl. The tallest is"," Anna",[" Ben"," Carl"," Dana"]),
 ("Anna is taller than Ben. Ben is taller than Carl. The shortest is"," Carl",[" Anna"," Ben"," Dana"]),
 ("The box is inside the bag. The bag is inside the car. The car contains the"," box",[" road"," driver"," key"]),
 ("Tom runs faster than Sam. Sam runs faster than Joe. The slowest is"," Joe",[" Tom"," Sam"," Ana"]),
 # syllogism
 ("Every cat is an animal. Tom is a cat. Tom is an"," animal",[" insect"," object"," idea"]),
 ("All birds have feathers. A robin is a bird. A robin has"," feathers",[" scales"," fur"," gills"]),
 ("No fish can fly. A salmon is a fish. A salmon cannot"," fly",[" swim"," eat"," move"]),
 ("All squares have four sides. This shape is a square, so it has four"," sides",[" corners of gold"," wheels"," colours"]),
 # negation
 ("The lamp is not on. The lamp is"," off",[" on"," bright"," new"]),
 ("The door is neither open nor locked. The door is"," closed",[" open"," locked"," painted"]),
 ("She is not older than him, and not the same age. She is"," younger",[" older"," taller"," wiser"]),
 ("The glass is not empty and not full. The glass is partly"," full",[" empty"," broken"," blue"]),
 # counting
 ("The word cat has how many letters? It has"," three",[" two"," four"," five"]),
 ("The word table has how many letters? It has"," five",[" four"," six"," seven"]),
 ("In the list red, blue, green there are how many colours? There are"," three",[" two"," four"," five"]),
 ("A week has seven days. Two weeks have"," fourteen",[" seven"," ten"," twenty"]),
 # induction: continue a pattern given earlier in the same prompt
 ("Pattern: red blue green red blue green red blue"," green",[" red"," blue"," yellow"]),
 ("Pattern: 2 4 6 8 10"," 12",[" 11"," 14"," 9"]),
 ("Pattern: a b a b a b a"," b",[" a"," c"," d"]),
 ("Pattern: Monday Tuesday Wednesday Monday Tuesday"," Wednesday",[" Monday"," Tuesday"," Thursday"]),
 ("The code word is zebra. Remember it. The code word was"," zebra",[" tiger"," lion"," horse"]),
 ("Kim's key is number 47. Later, Kim's key is number"," 47",[" 48"," 74"," 41"]),
 # units and quantity
 ("One hour has sixty"," minutes",[" seconds"," days"," weeks"]),
 ("One kilometre is one thousand"," metres",[" centimetres"," millimetres"," kilograms"]),
 ("Half of ten is"," five",[" two"," twenty"," fifteen"]),
 ("Double four is"," eight",[" two"," six"," fourteen"]),
 # everyday cause and effect
 ("It rained all night, so the road in the morning was"," wet",[" dry"," dusty"," empty"]),
 ("He forgot to water the plant, so the plant"," died",[" grew"," sang"," flew"]),
 ("The shop was closed, so she could not"," enter",[" sleep"," breathe"," think"]),
 ("The ice was left in the sun, so it"," melted",[" froze"," grew"," burned"]),
    ]
