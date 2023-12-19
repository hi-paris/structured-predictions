# pythonOK3
classes and methods to implement the OK3 method : decision trees with a kernelized output for structured prediction.


Afin de tester facilement OK3, j'ai fais en sorte qu'il puisse prendre en argument des matrices de vecteurs de sortie plutôt que uniquement des matrices de Gram, il construira ensuite en interne la matrice de Gram des sorties et la donnera au "vraies" méthodes fit et predict notamment.


Pour tester les arbres OK3 pour de la classification multilabel ou de la régression sur des données vectorielles (avec respectivement un critère d'impurté Gini et variance), une série de tests est rédigée dans le fichier tests/test_tree_clf_and_reg.py.



Protocole :


1 - Cloner le projet 'pythonOK3'


2 - Dans son dossier cloné, exécuter dans un terminal : python setup.py build_ext --inplace

	Cela va compiler les différents fichiers Cython, et certainment lever plusieurs warnings (à ignorer)


3 - Eventuellement restart le kernel de la console iPython si l'on souhaite y travailler pour prendre en compte les chanements dans les fichiers compilés


4 - Pour lancer la batterie de tests, entrer dans un terminal : pytest tests/test_treeclf_and_reg.py

	Cela va lancer la totalité des tests de ce fichier, certains étant assez longs. (Durée totale inférieure à 11min)
	Ces tests sont appliqués à des tâches de régression et de classification uniquement, il reste à tester sur des vrais problèmes structurés.


5 - Une autre manière de tester très rapidement ces fonctions de classification et régression et de comparer aux résultats obtenus avec les arbres de classification et régression classique est d'exécuter les fichiers test_classification et test_regression qui vont imprimer certaines lignes démontrant la quasi identité des arbres construits par OK3 et ceux classiques.


6 - Pour tester la prédiction structurée (différent de la régression et classification simple), voir le fichier tests.exemple_utilisation.py qui décrit comment utiliser les arbres ok3 (sur un problème de classification multilabel pouvant s'apparenter à un pb de prédiction structurée).

