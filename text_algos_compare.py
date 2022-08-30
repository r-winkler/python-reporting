import textdistance
from tabulate import tabulate


print()
reference = input("Enter a reference string:\n")
# reference = "rosser"
# search = "rossi"

while True:

    print()
    search = input("Enter search:\n")


    print(tabulate([
        ['distance', textdistance.hamming.distance(search, reference), textdistance.levenshtein.distance(search, reference), textdistance.damerau_levenshtein.distance(search, reference), textdistance.jaro.distance(search, reference), textdistance.jaro_winkler.distance(search, reference), textdistance.mlipns.distance(search, reference), textdistance.strcmp95.distance(search, reference), textdistance.needleman_wunsch.distance(search, reference), textdistance.gotoh.distance(search, reference), textdistance.smith_waterman.distance(search, reference)],
        ['similarity', textdistance.hamming.similarity(search, reference), textdistance.levenshtein.similarity(search, reference), textdistance.damerau_levenshtein.similarity(search, reference), textdistance.jaro.similarity(search, reference), textdistance.jaro_winkler.similarity(search, reference), textdistance.mlipns.similarity(search, reference), textdistance.strcmp95.similarity(search, reference), textdistance.needleman_wunsch.similarity(search, reference), textdistance.gotoh.similarity(search, reference), textdistance.smith_waterman.similarity(search, reference)],
        ['normalized distance', textdistance.hamming.normalized_distance(search, reference), textdistance.levenshtein.normalized_distance(search, reference), textdistance.damerau_levenshtein.normalized_distance(search, reference), textdistance.jaro.normalized_distance(search, reference), textdistance.jaro_winkler.normalized_distance(search, reference), textdistance.mlipns.normalized_distance(search, reference), textdistance.strcmp95.normalized_distance(search, reference), textdistance.needleman_wunsch.normalized_distance(search, reference), textdistance.gotoh.normalized_distance(search, reference), textdistance.smith_waterman.normalized_distance(search, reference)],
        ['normalized similarity', textdistance.hamming.normalized_similarity(search, reference), textdistance.levenshtein.normalized_similarity(search, reference), textdistance.damerau_levenshtein.normalized_similarity(search, reference), textdistance.jaro.normalized_similarity(search, reference), textdistance.jaro_winkler.normalized_similarity(search, reference), textdistance.mlipns.normalized_similarity(search, reference), textdistance.strcmp95.normalized_similarity(search, reference), textdistance.needleman_wunsch.normalized_similarity(search, reference), textdistance.gotoh.normalized_similarity(search, reference), textdistance.smith_waterman.normalized_similarity(search, reference)],
        ['distance (qval=2)', textdistance.Hamming(qval=2).normalized_distance(search, reference), textdistance.Levenshtein(qval=2).normalized_distance(search, reference), textdistance.DamerauLevenshtein(qval=2).normalized_distance(search, reference), textdistance.Jaro(qval=2).normalized_distance(search, reference), textdistance.JaroWinkler(qval=2).normalized_distance(search, reference), textdistance.MLIPNS(qval=2).normalized_distance(search, reference), 'n/a', textdistance.NeedlemanWunsch(qval=2).normalized_distance(search, reference), textdistance.Gotoh(qval=2).normalized_distance(search, reference), textdistance.SmithWaterman(qval=2).normalized_distance(search, reference)]
    ],
        headers=['', 'Hamming', 'Levenshtein', 'Damerau-Levenshtein', 'Jaro', 'Jaro-Winkler', 'MLIPNS', 'STRCMP95', 'Needleman-Wunsch', 'Gotoh', 'Smith-Waterman']))

exit()
