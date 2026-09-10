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
