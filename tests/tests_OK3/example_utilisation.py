######## Status ###########
## File is executed withou error but some warnings
## use below command
## python tests/tests_OK3/exemple_utilisation.py


############################################################
# # # # #   Exemple d'utilisation d'un arbre OK3   # # # # #
############################################################

from stpredictions.models.OK3._classes import OK3Regressor, ExtraOK3Regressor
from stpredictions.models.OK3.kernel import *
from sklearn import datasets

# %% Générer un dataset dont les sorties sont des longs vecteurs de 0 et de 1

n_samples = 4000
n_features = 100
n_classes = 1000
n_labels = 2

# On va fitter sur la moitié, 
#       tester sur un quart 
#    et se servir du dernier quart comme ensemble de sorties possibles

# C'est un gros jeu de données pour l'algorithme, qui montre un peu de lenteur,
# il faut surtout bien penser à mettre des paramètres régulant la croissance de l'abre :
# --> max_depth, max_features 

X, y = datasets.make_multilabel_classification(n_samples=n_samples,
                                               n_features=n_features,
                                               n_classes=n_classes,
                                               n_labels=n_labels)

# La première moitié constitue les données d'entrainement
X_train = X[:n_samples // 2]
y_train = y[:n_samples // 2]

# Le troisième quart les données de test
X_test = X[n_samples // 2: 3 * n_samples // 4]
y_test = y[n_samples // 2: 3 * n_samples // 4]

# Le dernier quart des sorties est utilisé pour fournir des candidats pour le décodage de l'arbre
# Les prédictions seront donc dans cet ensemble
y_candidates = y[3 * n_samples // 4:]

# %% Fitter un (deux) arbre(s) aux données

# Pour cela il faut renseigner un noyau à utiliser sur 
# les données de sorties sous forme vectorielles ci-dessus.

# Pour l'instant la liste des noyaux utilisables est :
kernel_names = ["linear", "mean_dirac", "gaussian", "laplacian"]

# Les noyaux gaussiens et exponentiels ont un paramètre gamma réglant la "largeur" du noyau


# Choisissons un noyau
# On peut indifféremment renseigner le nom (et les éventuels paramètres) :
kernel1 = ("gaussian", .1)  # ou bien kernel1 = "linear"
# ou bien
kernel2 = Mean_Dirac_Kernel()

# Ensuite on peut créer notre estimateur, qui travaillera en sachant calculer des noyaux gaussiens entre les sorties :
ok3 = OK3Regressor(max_depth=6, max_features='sqrt', kernel=kernel1)

# c'est également à la création de l'estimateur que l'on peut renseigner 
# la profondeur maximale, la réduction minimale d'impureté à chaque split, 
# le nombre minimal de sample dans chaque feuille, etc comme pour les arbres classiques

# On peut maintenant fitter à nos données :
ok3.fit(X_train, y_train)
# on aurait pu également renseigner un paramètre sample_weight : vecteur de 
# poids positifs ou nuls sur les exemples d'entrainement : un poids zéro sur
# un exemple signifie que l'exemple ne sera pas pris en compte
print("check")
# ALTERNATIVE : on peut ne renseigner le mode de calcul des noyaux que lors du 'fit' 
# ce qui permet de changer de noyau avec le même estimateur
# ex:
extraok3 = ExtraOK3Regressor(max_depth=6, max_features='sqrt')
extraok3.fit(X_train, y_train, kernel=kernel2)  # l'estimateur garde en mémoire ce nouveau kernel

# %% (OPTIONNEL) Avant de décoder l'arbre

# On peut :
# obtenir la profondeur de l'arbre
depth = ok3.get_depth()
# obtenir le nombre de feuilles
n_leaves = ok3.get_n_leaves()
# obtenir les prédictions de chaque feuille sous forme de vecteurs de poids sur les sorties d'entrainement
leaves_preds_as_weights = ok3.get_leaves_weights()
# obtenir les indices des feuilles dans lesquelles tombent de nouvelles données
X_test_leaves = ok3.apply(X_test)
# obtenir les prédictions sous forme de poids pour de nouvelles données
test_weights = ok3.predict_weights(X_test)
# calculer le score R2 de nos prédictions dans l'espace de Hilbert dans lequel sont plongées les sorties
r2_score = ok3.r2_score_in_Hilbert(X_test, y_test)  # on peut y spécifier un vecteur sample_weight

# %% Décodage de(s) (l')arbre(s) pour la prédiction des sorties

# Pour le décodage, on peut proposer un ensemble de sorties candidates, 
# ou bien ne rien fournir (None), dans ce cas les sorties candidates 
# sont les sorties des exemples d'entrainement.
candidates_sets = [None, y_candidates]
# on peut essayer avec candidates=y_candidates ou bien avec rien (ou None)

# On peut soit décoder l'arbre par rapport à un ensemble de candidats de sorties
leaves_preds = ok3.decode_tree(candidates=y_candidates)
# Puis pouvoir prédire des sorties directement
y_pred_1 = ok3.predict(X_test[::2])
y_pred_2 = ok3.predict(X_test[1::2])

print("check")

# Soit décoder pour une série d'entrées en précisant les candidats
y_pred_extra_1 = extraok3.predict(X_test[::2], candidates=y_candidates)
y_pred_extra_2 = extraok3.predict(X_test[1::2])  # on se souvient des prédictions de chaque feuille

# %% Evaluation des performances

# On peut calculer le score R2 dans l'espace de Hilbert comme dit précédemment (qui ne nécessite pas de décodage):
r2_score = extraok3.r2_score_in_Hilbert(X_test, y_test)  # on peut y spécifier un vecteur sample_weight

# On peut calculer des scores sur les vraies données décodées :
hamming_score = ok3.score(X_test, y_test, metric="hamming")  # on peut y spécifier un vecteur sample_weight
# remarque : on est pas obligé de repréciser un ensemble de candidats maintenant que ok3 en a déjà reçu un

# Une des métrique disponible pour cette fonction score est la présence de la 'top k accuracy'.
# Pour cela il faut renseigner l'ensemble des candidats dans la fonction à chaque appel:
top_3_score = ok3.score(X_test, y_test, candidates=y_candidates, metric="top_3")
# on peut renseigner n'importe quel entier à la place du 3 ci-dessus :
top_11_score = ok3.score(X_test, y_test, candidates=y_candidates, metric="top_11")

# %% A savoir

##### A propos de 'candidates' #####

# Il faut savoir que si lorsqu'ils sont requis, les ensembles de candidats ne sont pas fournis, 
# alors la recherche se fait dans l'ensemble des sorties du dataset d'entrainement, qui est 
# mémorisé par l'arbre (plus rapide).

# Les condidats peuvent être renseignés dans les fonctions 'decode_tree', 'predict', 
# 'decode' (qui est un synonyme de predict), et dans 'score'.
# Une fois un ensemble de candidats renseigné dans une de ces fonctions, l'ensemble des 
# prédictions possibles de ses feuilles est mémorisé par l'estimateur (mais pas 
# l'ensemble candidates lui même car étant potentiellement très grand), c'est pourquoi 
# il n'est pas nécessaire et même pas souhaitable computationnellement de renseigner 
# plusieurs fois le même ensemble de candidats, la première fois suffit. 
# En revanche lorsque l'on calcule un score de top k accuracy alors il faut
# obligatoirement renseigner un ensemble de candidats car le décodage est différent 
# si l'on doit renvoyer plusieurs prédictions pour chaque feuille.
