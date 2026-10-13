"""MANY KINDS OF FACT, shared by eval/build_recall.py (RECALL_KINDS=many) and eval/selfimprove.py (SI_KINDS=many).
2026-10-12. The six kinds of eval/build_recall.py taught six template-bound lookups: on two kinds no
builder writes (rivers into seas, guild leaders; eval/incontext.py INCONTEXT_KINDS=new) the base and the self-improved model
are at chance (24-35%), even with 10 facts in plain view, while the trained kinds reach 97%. Here ~400 kinds, built from
attributes of typed things and from relations of people, each in several wordings, so what is learned is "find the sentence
about this name", not six patterns. A document draws its facts from 1-6 kinds, many names per kind, so a question can only
be answered by the name, never by the relation alone. HELD OUT, never written here: rivers, seas, flowing, guilds, leaders /
leading, bridges, building, fruit (eval/incontext.py's new kinds and the diagnostic kinds).
"""

COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "grey", "brown", "pink", "silver"]

def many_kinds(name, r):
    """name(n): an invented name of n syllables; r: the caller's random.Random. Returns (KINDS, value): KINDS a list of
    (fact wordings, question wordings, value type) over {X} and {Y}; value(k) draws a value for kind k."""
    ANIMALS = ["fox", "owl", "bear", "wolf", "horse", "goat", "eagle", "otter", "badger", "heron", "lynx", "crane", "hare", "seal", "falcon", "beaver"]
    METALS = ["copper", "tin", "iron", "silver", "gold", "zinc", "nickel", "bronze", "brass"]
    INSTR = ["violin", "flute", "drum", "harp", "piano", "trumpet", "cello", "guitar", "oboe", "horn", "lute", "clarinet"]
    SPORTS = ["tennis", "rowing", "chess", "archery", "fencing", "cricket", "hockey", "swimming", "boxing", "cycling", "skiing", "golf"]
    FOODS = ["bread", "rice", "cheese", "barley", "beans", "honey", "salt", "wool", "timber", "oats", "fish", "olives"]
    GEMS = ["ruby", "emerald", "sapphire", "opal", "pearl", "amber", "jade", "topaz", "garnet", "onyx"]
    JOBS = ["baker", "carpenter", "nurse", "pilot", "farmer", "teacher", "weaver", "potter", "sailor", "miner", "tailor", "doctor", "judge", "clerk"]
    SUBJ = ["history", "music", "law", "medicine", "botany", "geometry", "poetry", "chemistry", "astronomy", "painting", "economics", "logic"]
    SHAPES = ["circle", "square", "triangle", "star", "crescent", "diamond", "cross", "spiral", "arrow", "crown"]
    COLLECT = ["stamps", "coins", "shells", "maps", "keys", "clocks", "buttons", "feathers", "postcards", "spoons"]
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    VT = {"person": (lambda: f"{name(2)} {name(2)}", "person"), "place": (lambda: name(2), "place"),
          "num": (lambda: str(r.randint(100, 9999)), "number"), "year": (lambda: str(r.randint(1200, 2020)), "year"),
          "colour": (lambda: r.choice(COLOURS), "colour"), "animal": (lambda: r.choice(ANIMALS), "animal"),
          "metal": (lambda: r.choice(METALS), "metal"), "food": (lambda: r.choice(FOODS), "product"),
          "gem": (lambda: r.choice(GEMS), "stone"), "shape": (lambda: r.choice(SHAPES), "shape"),
          "month": (lambda: r.choice(MONTHS), "month"), "word": (lambda: name(2), "name")}
    PLACES = ["country", "city", "town", "village", "island", "valley", "province", "district", "harbour", "kingdom"]
    BUILT = ["castle", "tower", "temple", "school", "library", "museum", "hospital", "theatre", "palace", "monastery", "lighthouse", "fortress"]
    ORGS = ["company", "club", "band", "choir", "team", "society", "orchestra", "academy", "bank", "newspaper"]
    WORKS = ["book", "song", "film", "poem", "painting", "opera", "play", "statue"]
    SKY = ["planet", "star", "comet", "moon"]
    VEH = ["ship", "train", "airship", "submarine"]
    NATURE = ["mountain", "lake", "forest", "desert", "glacier", "volcano", "cave", "canyon"]
    ATTRS = [  # (attribute, value type, the kinds of thing that have it)
        ("mayor", "person", ["city", "town", "village", "district"]), ("governor", "person", ["province", "island", "district"]),
        ("king", "person", ["kingdom", "island"]), ("population", "num", PLACES), ("national animal", "animal", ["country", "kingdom"]),
        ("main export", "food", PLACES), ("chief metal", "metal", ["country", "province", "valley", "kingdom"]),
        ("flag colour", "colour", ["country", "kingdom", "province", "city"]), ("emblem", "shape", PLACES + ORGS),
        ("founding year", "year", PLACES + BUILT + ORGS), ("founder", "person", ["city", "town", "village", "kingdom"] + BUILT + ORGS),
        ("architect", "person", BUILT), ("height in metres", "num", ["tower", "lighthouse", "statue", "mountain", "volcano"]),
        ("number of rooms", "num", ["castle", "palace", "hospital", "school", "monastery"]), ("keeper", "person", ["lighthouse", "museum", "library", "castle"]),
        ("head teacher", "person", ["school", "academy"]), ("chief doctor", "person", ["hospital"]), ("owner", "person", ORGS + BUILT + VEH),
        ("manager", "person", ["company", "team", "bank", "theatre", "club"]), ("captain", "person", VEH + ["team", "club"]),
        ("coach", "person", ["team", "club"]), ("conductor", "person", ["orchestra", "choir", "train"]),
        ("singer", "person", ["band", "choir", "song", "opera"]), ("editor", "person", ["newspaper"]),
        ("number of members", "num", ORGS), ("team colour", "colour", ["team", "club", "society", "academy"]),
        ("mascot", "animal", ["team", "club", "school", "academy", "company"]), ("author", "person", ["book", "poem", "play"]),
        ("composer", "person", ["song", "opera"]), ("director", "person", ["film", "play", "museum", "theatre", "bank"]),
        ("painter", "person", ["painting"]), ("sculptor", "person", ["statue"]), ("year of release", "year", ["book", "song", "film", "opera"]),
        ("hero", "person", ["book", "film", "play", "poem", "opera"]), ("discoverer", "person", SKY + ["island", "cave", "glacier"]),
        ("year of discovery", "year", SKY + ["island", "cave", "glacier", "volcano"]), ("colour", "colour", SKY + VEH + BUILT + ["painting"]),
        ("home port", "place", ["ship", "submarine", "airship"]), ("first stop", "place", ["train", "airship"]),
        ("length in metres", "num", VEH + ["cave", "canyon", "glacier"]), ("nearest town", "place", NATURE + BUILT),
        ("first climber", "person", ["mountain", "volcano", "glacier"]), ("depth in metres", "num", ["lake", "cave", "canyon"]),
        ("sacred stone", "gem", ["temple", "monastery", "kingdom", "mountain"]), ("festival month", "month", PLACES + ["temple", "monastery"]),
        ("patron", "person", ["temple", "monastery", "theatre", "orchestra", "academy", "library"]), ("old name", "word", PLACES + NATURE + SKY + VEH),
        ("serial number", "num", VEH), ("guardian animal", "animal", ["temple", "forest", "mountain", "castle"])]
    def attr_kind(at, vt, ty):
        sub = f"the {ty} {{X}}"; noun = VT[vt][1]
        facts = [f"The {at} of {sub} is {{Y}}.", f"{{Y}} is the {at} of {sub}.", f"{sub[0].upper() + sub[1:]} has {{Y}} as its {at}."]
        qs = [f"What is the {at} of {sub}?", f"Name the {at} of {sub}.",
              (f"Who is the {at} of {sub}?" if vt == "person" else f"Which {noun} is the {at} of {sub}?")]
        return (facts, qs, vt)
    KINDS = [attr_kind(at, vt, ty) for at, vt, tys in ATTRS for ty in tys]
    KINDS += [  # relations of people (subject a person's name, X)
        (["{X} plays the {Y}.", "The instrument {X} plays is the {Y}.", "{X} is a {Y} player."], ["Which instrument does {X} play?", "What does {X} play?", "Name the instrument of {X}."], "instr"),
        (["{X} works as a {Y}.", "{X} is a {Y} by trade.", "The job of {X} is {Y}."], ["What is the job of {X}?", "What does {X} work as?", "Name the trade of {X}."], "job"),
        (["{X} is married to {Y}.", "{Y} is the spouse of {X}.", "{X} and {Y} are married."], ["Who is {X} married to?", "Who is the spouse of {X}?", "Name the spouse of {X}."], "person"),
        (["{X} lives in {Y}.", "The home of {X} is in {Y}.", "{X} has a house in {Y}."], ["Where does {X} live?", "In which place is the home of {X}?", "Name the home town of {X}."], "place"),
        (["{X} keeps a pet {Y}.", "The pet of {X} is a {Y}.", "{X} owns a {Y} as a pet."], ["What pet does {X} keep?", "Which animal is the pet of {X}?", "Name the pet of {X}."], "animal"),
        (["{X} is {Y} years old.", "The age of {X} is {Y}.", "{X} turned {Y} this year."], ["How old is {X}?", "What is the age of {X}?", "Give the age of {X}."], "age"),
        (["{X} studied {Y}.", "The subject of {X} was {Y}.", "{X} has a degree in {Y}."], ["What did {X} study?", "Which subject did {X} study?", "Name the subject of {X}."], "subj"),
        (["{X} collects {Y}.", "The hobby of {X} is collecting {Y}.", "{X} has a large collection of {Y}."], ["What does {X} collect?", "Which things does {X} collect?", "Name the collection of {X}."], "collect"),
        (["The favourite sport of {X} is {Y}.", "{X} loves {Y} more than any sport.", "{X} plays {Y} every week."], ["What is the favourite sport of {X}?", "Which sport does {X} love?", "Name the sport of {X}."], "sport"),
        (["{X} was taught by {Y}.", "The teacher of {X} was {Y}.", "{Y} taught {X}."], ["Who taught {X}?", "Who was the teacher of {X}?", "Name the teacher of {X}."], "person"),
        (["{X} sleeps in room {Y}.", "The room of {X} is number {Y}.", "Room {Y} belongs to {X}."], ["Which room does {X} sleep in?", "What is the room number of {X}?", "Give the room of {X}."], "num"),
        (["{X} wears a {Y} coat.", "The coat of {X} is {Y}.", "{X} always wears {Y}."], ["What colour is the coat of {X}?", "Which colour does {X} wear?", "Name the colour of the coat of {X}."], "colour"),
        (["{X} was born in the year {Y}.", "The birth year of {X} is {Y}.", "{X} came into the world in {Y}."], ["In which year was {X} born?", "What is the birth year of {X}?", "Give the year {X} was born."], "year"),
        (["{X} wrote the book called {Y}.", "The book {Y} is by {X}.", "{X} is the writer of {Y}."], ["Which book did {X} write?", "What is the book by {X} called?", "Name the book of {X}."], "word"),
        (["The best friend of {X} is {Y}.", "{Y} is the closest friend of {X}.", "{X} and {Y} are best friends."], ["Who is the best friend of {X}?", "Who is the closest friend of {X}?", "Name the friend of {X}."], "person"),
        (["{X} owns a {Y} ring.", "The ring of {X} is set with a {Y}.", "{X} wears a ring of {Y}."], ["Which stone is in the ring of {X}?", "What is the ring of {X} set with?", "Name the stone of {X}."], "gem"),
        (["{X} travels in {Y} every year.", "Each year in {Y}, {X} goes on a journey.", "The travel month of {X} is {Y}."], ["In which month does {X} travel?", "When does {X} go on a journey?", "Name the travel month of {X}."], "month"),
        (["{X} was elected in {Y}.", "The election of {X} took place in {Y}.", "{X} won the vote in {Y}."], ["When was {X} elected?", "In which year was {X} elected?", "Give the election year of {X}."], "year")]
    VT.update({"instr": (lambda: r.choice(INSTR), "instrument"), "job": (lambda: r.choice(JOBS), "job"),
               "age": (lambda: str(r.randint(5, 99)), "age"), "subj": (lambda: r.choice(SUBJ), "subject"),
               "collect": (lambda: r.choice(COLLECT), "things"), "sport": (lambda: r.choice(SPORTS), "sport")})
    BAN = ("river", "sea", "flow", "guild", "lead", "bridge", "built by", "build", "fruit")
    assert not any(b in " ".join(f + q) .lower() for f, q, _ in KINDS for b in BAN), "a held-out word in a training kind"
    def value(k): return VT[KINDS[k][2]][0]()
    return KINDS, value
