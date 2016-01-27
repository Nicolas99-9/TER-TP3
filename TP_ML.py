from __future__ import division

import collections
import operator
import numpy as np
import operator
from pprint import pprint

# Travail sur les donnees:
# On importe le corpus movie_reviews de nltk
from nltk.corpus import movie_reviews
# On recupere la liste des fichiers: chaque fichier contient une critique.
# Les 1000 premieres sont negatives, les 1000 suivantes sont positives.
Ids = movie_reviews.fileids()

# Creer une fonction count_words qui prend en entree une liste de mots et retourne un dictionnaire de comptes
# On pourra utiliser la fonction Counter du module collections 

def count_words(words):
    return collections.Counter(words)


#print(count_words(Ids))
# Creer une fonction combine_counts qui prend en entree une liste de dictionnaires et retourne un dictionnaire qui les combine
# Ses valeurs sont la somme des valeurs des dictionnaires d'entree 

def combine_counts(counts):
   mon_dico = {}
   for dicos in counts:
       for value in dicos:
           if mon_dico.has_key(value):
               mon_dico[value] += dicos[value]
           else:
               mon_dico[value] = dicos[value]
   return mon_dico

#print(combine_counts([{'a': 1,'b' :3}, {'a': 1,'c' :2}]) )      
# Creer une fonction get_n_top_words qui prend en entree un dictionnaire et un entier n 
# et qui retourne la liste des n elements les plus frequents du dictionnaire
# On pourra utiliser la fonction itemgetter du module operator       

def get_n_top_words(count, n):
    return collections.Counter(count).most_common(n)

#print(get_n_top_words({'a': 1,'b' :3, 'd' : 1 , 'e' : 98}, 2 ) )     
# Creer une fonction get_top_values qui prend en entree un dictionnaire et une liste d'elements                                                                                                             
# et renvoie la liste des valeurs associees a ces elements (dans le meme ordre)                       

def get_top_values(count, top_keys):
    liste = []
    for s in top_keys:
        if s in count:
            liste.append(count[s])
        else:
            liste.append(0)
    return liste


#print(get_top_values({'a': 1,'b' :3, 'd' : 1 , 'e' : 98},['a','z','e']) )  
#

# Creer une fonction normalize_counts qui prend en entree un dictionnaire de comptes
# et qui renvoie ce dictionnaire avec des valeurs normalisees 

def normalize_counts(count):
    total = 0.0
    for s in count:
        total += count[s]
    for s in count:
        count[s] = count[s] / total
    return count
        
#print(normalize_counts({'a': 1,'b' :3, 'd' : 1 , 'e' : 4}))
# Combiner toutes ces fonctions dans une fonction get_counts_matrix qui prend en entree la liste Ids des noms de fichiers et un entier n
# et renvoie pour chacun d'entre eux la liste des n comptes normalises associes aux n mots les plus frequents de l'*ensemble* du corpus
# On peut recuperer les mots associes a un fichier fid sous forme de liste en utilisant la fonction
# movie_reviews.words(fileids=fid)
# On transformera la liste (iteree sur les fichiers) de liste de comptes en matrice a l'aide de la fonction
# numpy.array()

def get_counts_matrix(Ids, n):
    resultat = []
    dics = {}
    tmp = []
    for fid in Ids:
        liste_temp = movie_reviews.words(fileids=fid)
        count = count_words(liste_temp)
        dics[fid]  = count
        tmp.append(count)
    for e in dics:
         dics[e]  = normalize_counts(dics[e])
    dico_finale = combine_counts(tmp)
    dico_normalized = normalize_counts(dico_finale)
    top_words = get_n_top_words(dico_finale,n)
    for fich in dics:
        tab = []
        for mot in top_words:
            if dics[fich].has_key(mot[0]):
                tab.append(dics[fich][mot[0]])
            else:
                tab.append(0)
        resultat.append(tab)
    return resultat    
# Choisir n, et obtenir la matrice demandee pour le corpus movie_reviews.
# Verifier que la matrice a la taille attendu avec la methode shape

n = 2000
val = get_counts_matrix(Ids,n)
mon_resultat = np.array(val)
# ligne => fichier
M = mon_resultat
# PCA
# On utilise la classe PCA du module mlab de matplotlib
# Les donnees projetees selon les nouvelles coordonnees sont contenues dans l'attribut PCA.Y

# La fonction plot_PCA vise a afficher les deux premiere dimensions des donnees dans les nouvelles coordonnees
# Completer la fonction plot_PCA en donnant en abcisse les points correspondant a la premiere dimension
# et en ordonnee les points correspondant a la deuxieme, separement pour les critiques positives et negatives
# Ensuite, l'utiliser sur les donnes. 

# PCA : PCA de la matrice => cree un objet pour associer le PCA a nos objets
# obtenir les donnes projetes => MR_PCA.Y  (attribut Y contient les donnes, donnes reprojetes dans le bon domaine)
# donnes dans un nouveau repere avec n dimensions mais on veut reduire cette dimension a 2 pour ensuite les afficher
# PCA.y => premiere dimension le nombre de critiques et ensuite les 1000 critiques
# PCA : anlyse en composante principal



import matplotlib
matplotlib.use('Agg') #sauvegarde a distance
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
  
MR_PCA = PCA(M)

def plot_PCA(PCA_projection):  
    # 2 premieres dimensions des 100 premieres composantes (critiques positivies)
    # ids premieres axes, dimensions de la projection seconde coordonne
    x_neg = PCA_projection[1000:][0]
    y_neg = PCA_projection[1000:][1]
    x_pos = PCA_projection[:1000][0]
    y_pos = PCA_projection[:1000][1]
    #x neg commentaire negatifs
    plt.figure()
    plt.xlabel('First component')
    plt.xlabel('Second component')
    plt.title("PCA applied to texts of the Movie_reviews corpus")
    plt.plot(x_neg,y_neg,'ro')
    plt.plot(x_pos,y_pos,'bo')
    plt.savefig("PAC.eps",format='eps',dpi=1000)
    

#plot_PCA(MR_PCA.Y)
"""
"""

# Perceptron
# On utilise la classe Perceptron: elle comprend un constructeur lui fournit donnees et parametres
# Et deux methodes, pour l'entrainement, et pour le test
# Comprendre le code et completer la ligne manquante a l'aide de la procedure decrite dans les slides
# On evitera une boucle for en vectorialisant, c'est a dire en executant l'operation pour toutes les donnes
# a la fois a l'aide d'une operation matricielle.

class perceptron(object):
    def __init__(self, data, iterations, weights = None, learning_rate = 0.1):
        self.data = data
        self.it = iterations
        # Si on ne precise pas comment initialiser les parametres, le modele les met a zero
        if weights == None:
            self.weights = np.zeros(np.shape(data)[1])
        self.l_r = learning_rate

    def train(self, labels):
        counter = 0
        # On creer des variables pour retenir les meilleurs parametres
        best_weights = self.weights
        best_error_rate = len(labels)
        while (counter < self.it):
            # On utilise les parametres pour estimer les labels des donnes
            estimate_labels = -np.ones(len(labels))
            estimate_labels[np.dot(self.data, self.weights) > 0] = 1
            # On cree un tableau qui nous indiquera pour quels exemples notre modele a fait une erreur
            errors = np.arange(0,len(labels))[ labels != estimate_labels ]
            # Si il n'y a pas d'erreur, on arrete
            if not errors.size:
                break
            counter += 1
            # Completer le code pour mettre a jour l'attribut self.weights 
            self.weights += (np.dot(self.l_r,np.dot(self.data,labels)))/ counter
            # Si notre modele s'est ameliore, on garde en memoire ses parametres
            if (errors.size < best_error_rate):
                best_weights = self.weights
        # Une fois le nombre maximal d'iterations termine, on garde les meilleurs parametres
        self.weights = best_weights

    def test(self, labels):
        # On calcule la proportion d'erreurs sur les donnes
        estimate_labels = -np.ones(np.shape(labels))
        estimate_labels[np.dot(self.data, self.weights) > 0] = 1
        errors = np.ones(np.shape(labels))[ labels != estimate_labels ]
        return sum(errors)/len(labels)


# Une fois la classe completee, creer un vecteur contenant les labels, creer la classe associee a nos donnes,
# et entrainer et tester le perceptrons sur les labels.
"""
"""
tab = [0 for i in range(1000)]
for e in range(1000):
    tab.append(1)
print(len(tab)) 
y = tab  
model =  [0 for i in range(n)]
percepttron = perceptron(mon_resultat,500,None,0.1) 
print(percepttron.train(y))
print(percepttron.test(y))
"""
"""

# Regression logistique
# On utilise la classe logistic_regression: elle est presque identique a la classe perceptron.
# La difference dans la procedure se trouve au niveau de la mise a jour: on doit calculer le gradient 
# et les utiliser pour mettre a jour les parametres.
# Completer le code, ici aussi en vectorialisant les calculs.
"""
class logistic_regression(object):
    def __init__(self, data, iterations, weights = None ):
        self.data = data
        self.it = iterations
        if weights == None:
            self.weights = np.zeros(np.shape(data)[1])
        self.it_counter = 0

    def train(self, labels, learning_rate = 1.0):
        counter = 0
        while (counter < self.it):
            # Ici, calculer les log probabilite en utilisant les donnees et parametres
            #log_probabilities = #
            self.it_counter += 1
            counter +=1
            # Ici, calculer le gradient a l'aide de la formule des slides
            #gradient = #
            # Puis l'utiliser pour mettre a jour l'attribut self.weights
            #self.weights += #

    def test(self, labels):
        log_probabilities = 1 / (1 + np.exp(-np.dot(self.data, self.weights)))
        estimate_labels = np.round(log_probabilities)
        errors = np.ones(np.shape(labels))[ labels != estimate_labels ]
        return sum(errors)/len(labels)

# Une fois la classe completee, creer une vecteur contenant les labels (qui sont differents des precedents! Pourquoi ?)
# creer la classe associee a nos donnes, et tester
# On fera cette fois plus attention a la procedure: il faudra separer les donnes d'entrainement et de test,
# et verifier la progression du modele periodiquement
"""
"""
y = 
model =
"""

"""
# Comment evoluent les resultats si l'on utilise d'autres features ? On peut essayer d'utiliser des bigrams, des n-grams
# ou d'autres features plus specifique.
# Comment le PCA pourrait-il etre utile ici ? 
"""


