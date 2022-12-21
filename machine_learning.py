#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import mlxtend.feature_selection as fs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from multiprocessing import cpu_count
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.svm import LinearSVR, LinearSVC, SVC, SVR
from sklearn.cluster import KMeans

# ==== Constantes des ML ========
CLASSIFICATION = 0
REGRESSION = 1
# ==== Constantes des modèles ========
CART = 0
RANDOM_FOREST = 1
BAGGING = 2
BOOSTING = 3
SVM_1 = 4

# ====== Constantes de SVM =====
LINEAR_KERNEL = 0
POLYNOMIAL_KERNEL = 1
RBF_KERNEL = 2
SIGMOID_KERNEL = 3

ONE_VS_ONE = 0
ONE_VS_REST = 1


def charger_donnees(filename):
    if not os.path.exists(filename):
        raise Exception("Le chemin du fichier specifié n'existe pas! peut-être supprimer les guillements!")

    sep = int(input("separateur de fichier:\n0(tabulation)\n1:(virgule)\n2:(point virgule)\n"))
    while sep not in [0, 1, 2]:
        sep = int(input("separateur de fichier:\n0(tabulation)\n1:(virgule)\n2:(point virgule)\n"))

    if sep == 0:
        sep = "\t"
    elif sep == 1:
        sep = ","
    else:
        sep = ";"
    return pd.read_csv(filename, sep, header=0)


def display_classes_stats(df, col_index):
    if df is None:
        raise Exception("Aucune donnée n'a été chargée!!")

    stats = df[df.columns[col_index]].value_counts()
    print(stats)


def normaliser(data, type_norm=0):
    scaler = StandardScaler() if type_norm == 0 else MinMaxScaler()
    scaler = scaler.fit(data)
    return scaler.transform(data) if type_norm == 0 else scaler.transform(data)


def get_range(index):
    if ":" in index:
        tmp = index.split(":")
        if len(tmp) < 2:
            raise Exception("Le format des indices n'est pas correct")
        beg = int(tmp[0])
        end = int(tmp[1])
        step = 1
        if len(tmp) == 3:
            step = int(tmp[2])

        if end < beg:
            c = end
            end = beg
            beg = c
        return [i for i in list(np.arange(beg, end + 1, step))]
    elif "," in index:
        return [int(i) for i in index.split(",") if len(i.strip()) > 0]

def extract(dt):
    print('Information sur les attributs')
    for i, col in enumerate(dt.columns):
        print("{} : {}".format(col, i))
    try:
        cible_index = int(input(
            "Choisir l'index de la variable cible : \nNB : Dans le cas d'un apprentissage non supervisé, laisser ce champ vide\n"))
    except:
        cible_index = -1

    active_feature = input(
        "Sélectionner les indices des attributs (séparer par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser ':' (ex. 0:7): ")

    if ":" in active_feature:
        tmp = active_feature.split(":")
        if len(tmp) != 2:
            raise Exception("Le format des indices n'est pas correct")
        beg = int(tmp[0])
        end = int(tmp[1])
        if beg > len(dt.columns) or end > len(dt.columns):
            raise Exception("L'un des indices est supérieur au nombre d'attributs disponibles")
        if end < beg:
            c = end
            end = beg
            beg = c
        active_feature = [i for i in list(range(beg, end + 1))]
    elif "," in active_feature:
        active_feature = [int(i) for i in active_feature.split(",") if len(i.strip()) > 0]
    else:
        try:
            beg = int(active_feature)
            active_feature = [beg]
        except Exception as ex:
            raise ex

    active_feature = list(filter(lambda elt: elt != cible_index and elt < len(dt.columns), active_feature))
    active_feature.sort()

    y = dt.iloc[:, cible_index]
    x = dt.iloc[:, active_feature]
    return cible_index, active_feature, x, y


def split_train_test_data(x, y, train_size=0.8):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, train_size=train_size)
    return Xtrain, Xtest, Ytrain, Ytest


def build_cart(type_prediction):
    critere = get_criterion(type_prediction)
    try:
        max_depth = int(
            input("Specifier la profondeur maximale de l'arbre\nNB : laisser ce champ vide si non définie\n"))
    except:
        max_depth = None

    return CartRegressor(critere, max_depth) if type_prediction else CartClassifier(critere, max_depth)


def build_boosting(type_prediction):
    try:
        n_estimators = int(
            input("Spécifier le nombre d'arbre : \nNB : laisser ce champ vide si non définie (par defaut = 50)\n"))
    except:
        n_estimators = 50

    return BoostingRegressor2(n_estimators) if type_prediction else BoostingClassifier2(n_estimators)


def build_bagging(type_prediction):
    try:
        max_samples = float(input(
            "Entrer la taille de l'echantillon :\nNB : laisser ce champ vide si non définie (par defaut = 1.0)\n "))
    except:
        max_samples = 1.0
    try:
        max_features = float(input(
            "Entrer le nombre d’attributs tirés aléatoirement : \nNB : laisser ce champ vide si non définie (par defaut = 1.0)\n"))
    except:
        max_features = 1.0
    try:
        n_estimators = int(
            input("Spécifier le nombre d'arbre : \nNB : laisser ce champ vide si non définie (par defaut = 10)\n"))
    except:
        n_estimators = 10

    return BaggingRegressor2(max_samples, max_features, n_estimators) if type_prediction else BaggingClassifier2(max_samples, max_features, n_estimators)


def get_criterion(type_prediction):
    if type_prediction:  # regression
        criterion = ['mse', 'mae']
        try:
            critere = int(
                input("Selectionner le critére:\n0: mse (EQM)\n1: mae (Erreur Moyenne Absolue) (default = mse)\n"))
            critere = criterion[critere]
        except Exception as ex:
            critere = 'mse'
    else:
        criterion = ['gini', 'entropy']
        try:
            critere = int(input("Selectionner le critére:\n0: gini\n1: entropy (default = gini)\n"))
            critere = criterion[critere]
        except Exception as ex:
            critere = 'gini'
    return critere


def build_randomforest(type_prediction):
    try:
        n_estimators = int(
            input("Spécifier le nombre d'arbre : \nNB : laisser ce champ vide si non définie (par defaut = 10)\n"))
    except:
        n_estimators = 10

    try:
        max_depth = int(
            input("Specifier la profondeur maximale de l'arbre\nNB : laisser ce champ vide si non définie\n"))
    except:
        max_depth = None

    critere = get_criterion(type_prediction)

    try:
        max_features = float(input(
            "Entrer le nombre d’attributs tirés aléatoirement : \nNB : laisser ce champ vide si non définie (par defaut = 'auto')\n"))
    except:
        max_features = 'auto'



    return RandomforestClassifier2(criterion=critere, max_depth=max_depth, n_estimators=n_estimators,
                                   max_features=max_features) if not type_prediction else RandomforestRegressor2(
        criterion=critere, max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)


def build_svm(type_prediction):
    try:
        noyau = int(input(
            "Choisir le type de noyau:\n0 : lineaire (SVM linéaire)\n1: polynomial\n2: rbf\n3 : sigmoid\n(defaut = linéaire)\n"))
    except:
        noyau = LINEAR_KERNEL

    try:
        multi_class = int(input(
            "Sélectionner la stratégie muti-classe à utiliser : \n0 : One against One\n1 : 'One Versus Rest'\n(defaut = One Versus Rest\n"))
    except:
        multi_class = ONE_VS_REST

    try:
        cost = float(
            input("Definir le parametre cout :\nNB : laisser ce champ vide si non définie (par defaut = 1.0)\n"))
    except:
        cost = 1.0

    return SVMClassifier(kernel=noyau, cost=cost, multi_class=multi_class) if not type_prediction else SVMRegressor(
        kernel=noyau, cost=cost, multi_class=multi_class)


def build_model(methode_ml, type_prediction):
    if methode_ml == RANDOM_FOREST:
        return build_randomforest(type_prediction)
    elif methode_ml == BAGGING:
        return build_bagging(type_prediction)
    elif methode_ml == BOOSTING:
        return build_boosting(type_prediction)
    elif methode_ml == SVM_1:
        return build_svm(type_prediction)

    return build_cart(type_prediction)


def feature_selection(modele, x_train, y_train, feature_names=[]):
    methode = input(
        "Choisissez la méthode de sélection de variables :\n0 : Backward selection\n1 : Forward selection\n (defaut =Forward selection)\n")
    try:
        methode = int(methode)
    except:
        methode = 1
    # methode = "backward" if methode == 0 else 'forward'

    cv = input("spécifier la valeur du CV (cross-validation)\n (defaut = 5)\n")
    try:
        cv = int(cv)
    except:
        cv = 5

    scoring = 'accuracy' if modele.type_pred == CLASSIFICATION else "neg_mean_squared_error"

    k_features = x_train.shape[1] if methode else 1

    feat_select = fs.SequentialFeatureSelector(modele.modele, k_features=k_features, forward=methode, scoring=scoring,
                                               cv=cv)
    feat_select = feat_select.fit(x_train, y_train, custom_feature_names=feature_names)

    fig1 = plot_sfs(feat_select.get_metric_dict(), kind='std_dev')

    # plt.ylim([0.8, 1])
    methode = "backward" if methode == 0 else 'forward'
    plt.title('Sequential {} Selection (w. StdDev)'.format(methode))
    plt.grid()
    plt.show()

    results = pd.DataFrame.from_dict(feat_select.get_metric_dict()).T
    return results[["feature_names", "avg_score"]]


# def build_kmeans():
#     try:
#         n_clusters = int(input("Spécifier le nombre de cluster (K) que vous souhaiteriez avoir \n(defaut=4)\n"))
#     except:
#         n_clusters = 4


class Kmeans(KMeans):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                 precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated',
                 algorithm='auto'):
        Kmeans.__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                        precompute_distances=precompute_distances, verbose=verbose, random_state=random_state,
                        copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
        self.X = None
        self.modele = None

    def launch(self, X):
        self.X = X
        self.modele = self.fit(X)

    def afficher_resultats(self):
        # centroid

        # cluster
        return 0

    def export_dataset(self):
        return 0


class Learner:
    def __init__(self, nom=" ", type_pred=0):
        self.nom = nom
        self.type_pred = type_pred
        self.modele = None
        self.feature_names = []

        self.X_train = None
        self.y_train = None

    """
    Cette fonction permet de savoir si un modèle a déjà été construit ou non
    """

    def has_model(self):
        return self.modele

    def create_model(self):
        print("A implémenter")

    def afficher_parametres(self):
        print("Méthode de ML : ", self.nom)
        # print("Type de prédiction :", self.afficher_type_prediction())

    def apprendre(self, X_train, y_train, feature_names=None):
        if feature_names is None:
            feature_names = []
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.modele.fit(X_train, y_train)

    def tester(self, X_test):
        return self.modele.predict(X_test)

    def predire(self, ind):
        return self.modele.predict(ind)

    def afficher_performance(self, X_test, Y_test):
        print("A implementer")

    def choisir_parametres_a_optimiser(self):
        print("A implementer !!!")

    def exporter_modele(self, chemin):
        if not self.modele:
            raise Exception("Aucun modèle n'a été construit!")
        date = datetime.now()
        filename = '{}/{}_{}_{}_{}.model'.format(chemin, self.nom, date.year, date.month, date.day)
        fp = open(filename, "wb")
        pickle.dump(self.modele, fp)

    def afficher_options_avancee_modele(self):
        print("A implementer")

    def afficher_modele(self):
        print("A implementer")

    def afficher_type_prediction(self):
        if self.type_pred == 0:
            return "Classification"
        else:
            return "Regression"

    def afficher_importance_attribut(self):
        print("A implémenter! !")

    def afficher_regle_decision(self):
        print("A implémenter! !")

    def validation_croise(self, X, Y, cv):
        scores = cross_val_score(self.modele, X, Y, cv=cv, n_jobs=cpu_count() - 1)
        return scores, scores.mean(), scores.var()

    def recherche_optimum(self):
        parameters = self.choisir_parametres_a_optimiser()
        print("====Paramètres à optimiser=======================")
        print("algorithme ML : ", self.nom)
        print(parameters)
        gridcv = GridSearchCV(self.modele, param_grid=parameters, n_jobs=cpu_count() - 1)
        gridcv.fit(self.X_train, self.y_train)
        return gridcv.best_params_, gridcv.best_score_

    def afficher_options_avancee_modele(self):
        print("Pas d'options disponible")


class Classifier(Learner):
    def __init__(self, nom=" "):
        Learner.__init__(self, nom, 0)

    def afficher_performance(self, X_test, Y_test):
        # accuracy = self.modele.score(X_test,Y_test)
        # print("Accuracy : {}".format(accuracy))

        Z = self.tester(X_test)
        print("=======Matrice de confusion ============ ")
        print(metrics.confusion_matrix(Y_test, Z))

        print("=======Autres indicateurs de performance ========== ")
        print(metrics.classification_report(Y_test, Z))


class Regressor(Learner):
    def __init__(self, nom=" "):
        Learner.__init__(self, nom, 1)

    def afficher_performance(self, X_test, Y_test):
        Z = self.tester(X_test)
        print("coefficient de détermination(R^2) : ", self.modele.score(X_test, Y_test))
        print("erreur quadratique moyenne (EQM) : ", metrics.mean_squared_error(Z, Y_test))


class Cart(object):
    def __init__(self, critere, max_depth):
        self.criterion = critere
        self.max_depth = max_depth
        self.model = None
        self.feature_names = None
        # self.create_model()

    def create_model(self):
        print("A implémenter ")

    def _afficher_modele(self):
        try:
            heigth = int(input("Saisir la hauteur du graphe (par defaut = 10)\n"))
        except:
            heigth = 10

        try:
            width = int(input("Saisir la largeur du graphe (par defaut = 10)\n"))
        except:
            width = 10
        plt.figure(figsize=(heigth, width))
        plot_tree(self.model, feature_names=self.feature_names, filled=True)

    def _afficher_importance_attribut(self):
        pd.set_option('display.max_rows', None)
        impVarFirst = {"Variable": self.feature_names, "Importance": self.model.feature_importances_}
        print(pd.DataFrame(impVarFirst).sort_values(by="Importance", ascending=False))

    def _afficher_regle_decision(self):
        tree_rules = export_text(self.model, feature_names=list(self.feature_names), show_weights=True)
        print(tree_rules)

    def options_avancee_modele(self):
        while True:
            options = input(
                "Quelle option souhaiteriez-vous afficher ?\n0 : Afficher le modèle\n1 : Afficher la règle de décision\n2 : Afficher l'importance des variables\n (defaut = Afficher le modèle)\n")
            try:
                options = int(options)
            except:
                options = 0
            if options == 0:
                self._afficher_modele()
                print("patienter ... (pour afficher le graphe, il est préférable de ne pas continuer (option = Non)")
                time.sleep(10)
            elif options == 1:
                self._afficher_regle_decision()
            else:
                self._afficher_importance_attribut()
            choix = input("Voulez vous continuer ?\n0 : Non\n1 : Oui (defaut = Oui)")
            try:
                choix = int(choix)
            except:
                choix = 1

            if choix == 0:
                break

    def _choisir_parametres_a_optimiser(self, type_prediction):
        max_depth = input(
            "spécifier les valeurs de max_depth : (séparer les valeurs par une virgule ex. 1,7,4,5,8)\nPour "
            "sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 0:7:0.1)")

        max_depth = get_range(max_depth)

        criterion = ["mse", "mae"] if type_prediction else ["gini", "entropy"]

        return {"criterion": criterion, "max_depth": max_depth}


class CartClassifier(Classifier, Cart):
    def __init__(self, critere, max_depth):
        Classifier.__init__(self, "Cart - Classifier")
        Cart.__init__(self, critere, max_depth)
        self.create_model()

    def create_model(self):
        self.modele = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Fonction de mesure d'impureté : ", self.criterion)
        print("Profondeur maximale :", self.max_depth)

    def afficher_options_avancee_modele(self):
        Cart.options_avancee_modele(self)

    def apprendre(self, X_train, y_train, feature_names=None):
        Learner.apprendre(self, X_train, y_train, feature_names)
        Cart.feature_names = feature_names

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser(0)


class CartRegressor(Regressor, Cart):
    def __init__(self, critere, max_depth):
        Regressor.__init__(self, "Cart - Regressor")
        Cart.__init__(self, critere, max_depth)
        self.create_model()

    def create_model(self):
        self.modele = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Fonction de mesure d'impureté : ", self.criterion)
        print("Profondeur maximale :", self.max_depth)

    def afficher_options_avancee_modele(self):
        Cart.options_avancee_modele(self)

    def apprendre(self, X_train, y_train, feature_names=None):
        Learner.apprendre(self, X_train, y_train, feature_names)
        Cart.feature_names = feature_names

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser(1)


class Bagging(object):
    def __init__(self, max_samples=1.0, max_features=1.0, n_estimators=10):
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.model = None
        self.create_model()

    def create_model(self):
        print("A implémenter")

    def afficher_modele(self):
        print("Pas de modele à afficher pour le bagging.Merci!")

    def _choisir_parametres_a_optimiser(self):
        n_estimators = input(
            "spécifier les valeurs du nombre d'arbres : (séparer les valeurs par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 10:1000:10)")

        n_estimators = get_range(n_estimators)
        return {"n_estimators": n_estimators}


class BaggingClassifier2(Classifier, Bagging):
    def __init__(self, max_samples=1.0, max_features=1.0, n_estimators=10):
        Classifier.__init__(self, "Bagging - Classifier")
        Bagging.__init__(self, max_samples=max_samples, max_features=max_features, n_estimators=n_estimators)

    def create_model(self):
        self.modele = BaggingClassifier(DecisionTreeClassifier(), max_samples=self.max_samples,
                                        max_features=self.max_features, n_estimators=self.n_estimators)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre maximal d'échantillons : ", self.max_samples)
        print("Nombre maximal d'attributs :", self.max_features)
        print("Nombre d'arbres :", self.n_estimators)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()


class BaggingRegressor2(Regressor, Bagging):
    def __init__(self, max_samples=1.0, max_features=1.0, n_estimators=10):
        Regressor.__init__(self, "Bagging - Regressor")
        Bagging.__init__(self, max_samples=max_samples, max_features=max_features, n_estimators=n_estimators)

    def create_model(self):
        self.modele = BaggingRegressor(DecisionTreeClassifier(), max_samples=self.max_samples,
                                       max_features=self.max_features, n_estimators=self.n_estimators)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre maximal d'échantillons : ", self.max_samples)
        print("Nombre maximal d'attributs :", self.max_features)
        print("Nombre d'arbres :", self.n_estimators)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()


class Randomforest(object):
    def __init__(self, criterion='gini', max_depth=None, n_estimators=100, max_features='auto'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.model = None
        self.create_model()

    def create_model(self):
        print("A implémenter")

    def afficher_model(self):
        print("Impossible d'afficher le modèle")

    def _choisir_parametres_a_optimiser(self, type_prediction):
        parameters = {}
        choix_criterion = input(
            "Souhaiteriez-vous utiliser la variable 'criterion(critère d'impureté)' comme paramètre à optimiser ?\n0 "
            ": Non\n1 : Oui\n(defaut = Oui)\n")
        try:
            choix_criterion = int(choix_criterion)
        except:
            choix_criterion = 1
        if choix_criterion:
            criterion = ["mse", "mae"] if type_prediction else ["gini", "entropy"]
            parameters["criterion"] = criterion

        n_estimators = input(
            "spécifier les valeurs du nombre d'arbres : (séparer les valeurs par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 10:1000:10)")

        parameters["n_estimators"] = get_range(n_estimators)

        choix_depth = input(
            "Souhaiteriez-vous utiliser la variable 'max_depth(profondeur de l'arbre)' comme paramètre à optimiser ?\n0 : Non\n1 : Oui\n(defaut = Oui)\n")
        try:
            choix_depth = int(choix_depth)
        except:
            choix_depth = 1
        if choix_depth:
            max_depth = input(
                "spécifier les valeurs de max_depth : (séparer les valeurs par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 2:20:2)")

            parameters["max_depth"] = get_range(max_depth)

        choix_feature = input(
            "Souhaiteriez-vous utiliser la variable 'max_feature(nombre d'attributs)' comme paramètre à optimiser ?\n0 : Non\n1 : Oui\n(defaut = Oui)\n")
        try:
            choix_feature = int(choix_feature)
        except:
            choix_feature = 1
        if choix_feature:
            # max_features = input(
            #     "spécifier les valeurs de max_features : (séparer les valeurs par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 2:20:2)")

            parameters["max_features"] = ['auto', 'sqrt', 'log2']

        return parameters


class RandomforestClassifier2(Classifier, Randomforest):
    def __init__(self, criterion='gini', max_depth=None, n_estimators=100, max_features='auto'):
        Classifier.__init__(self, "Randomforest - Classifier")
        Randomforest.__init__(self, criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                              max_features=max_features)

    def create_model(self):
        self.modele = RandomForestClassifier(self.n_estimators, max_depth=self.max_depth, criterion=self.criterion,
                                             max_features=self.max_features)
        self.model = self.modele

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser(0)

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre d'arbres :", self.n_estimators)
        print("Profondeur maximale des arbres : ", self.max_depth)
        print("Nombre maximal d'attributs :", self.max_features)
        print("Critère d'impureté :", self.criterion)


class RandomforestRegressor2(Regressor, Randomforest):
    def __init__(self, criterion='gini', max_depth=None, n_estimators=100, max_features='auto'):
        Regressor.__init__(self, "Randomforest - Regressor")
        Randomforest.__init__(self, criterion=criterion, max_depth=max_depth, n_estimators=n_estimators,
                              max_features=max_features)

    def create_model(self):
        self.modele = RandomForestRegressor(self.n_estimators, max_depth=self.max_depth, criterion=self.criterion,
                                            max_features=self.max_features)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre d'arbres :", self.n_estimators)
        print("Profondeur maximale des arbres : ", self.max_depth)
        print("Nombre maximal d'attributs :", self.max_features)
        print("Critère d'impureté :", self.criterion)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser(1)


class Boosting(object):
    def __init__(self, n_estimators):
        # self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = None
        self.create_model()

    def create_model(self):
        print("A implémenter !!")

    def afficher_model(self):
        print("Impossible d'afficher le modèle")

    def _choisir_parametres_a_optimiser(self):
        n_estimators = input(
            "spécifier les valeurs du nombre d'arbres : (séparer les valeurs par une virgule ex. 1,4,5,"
            "8)\nPour sélectionner plusieurs, utiliser ':' (ex. 0:7)")

        n_estimators = get_range(n_estimators)
        return {"n_estimators": n_estimators}


class BoostingClassifier2(Classifier, Boosting):
    def __init__(self, n_estimators=10):
        Classifier.__init__(self, "Boosting - Classifier")
        Boosting.__init__(self, n_estimators=n_estimators)

    def create_model(self):
        self.modele = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=self.n_estimators)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre d'arbres :", self.n_estimators)
        # print("Profondeur maximale :", self.max_depth)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()


class BoostingRegressor2(Regressor, Boosting):
    def __init__(self, max_samples=1.0, max_features=1.0, n_estimators=10):
        Regressor.__init__(self, "Boosting - Regressor")
        Boosting.__init__(self, n_estimators=n_estimators)

    def create_model(self):
        self.modele = AdaBoostRegressor(DecisionTreeClassifier(), n_estimators=self.n_estimators)
        self.model = self.modele

    def afficher_parametres(self):
        super().afficher_parametres()
        print("Nombre d'arbres :", self.n_estimators)
        # print("Profondeur maximale :", self.max_depth)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


class SVM(object):
    def __init__(self, kernel=RBF_KERNEL, cost=1, multi_class=ONE_VS_REST):
        self.cost = cost
        self.multi_class = "ovr" if multi_class == ONE_VS_REST else "crammer_singer"
        # self.dual=dual
        self.set_kernel(kernel)
        self.model = None
        self.create_model()

    def set_kernel(self, kernel):
        if kernel == LINEAR_KERNEL:
            self.kernel = "linear"
        elif kernel == RBF_KERNEL:
            self.kernel = "rbf"
        elif kernel == POLYNOMIAL_KERNEL:
            self.kernel = "poly"
        elif kernel == SIGMOID_KERNEL:
            self.kernel = "sigmoid"

    def create_model(self):
        print("A implémenter !!!")

    #         if self.type_pred == 0:
    #             self.modele=SVC(kernel=self.kernel,C=self.cost,decision_function_shape=self.multi_class)
    #         else:
    #             self.modele=SVR(kernel=self.kernel,C=self.cost,decision_function_shape=self.multi_class)

    # print("Optimisation duale de la marge ? :", self.dual)

    # def afficher_modele(self):
    #     print("Warning !!! l'affichage de l'hyperplan ne concerne que 2 attributs")
    #     index1 = int(input(""))
    #     index2 = 0
    #
    #     X0, X1 = self.X_train[:, index1], X_train[:, index2]
    #     xx, yy = make_meshgrid(X0, X1)
    #
    #     plot_contours(ax, clf, xx, yy,
    #                   cmap=plt.cm.coolwarm, alpha=0.8)
    #     ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel(self.feature_names[index1])
    #     ax.set_ylabel(self.feature_names[index2])
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     # ax.set_title(title)
    #     plt.show()

    def _choisir_parametres_a_optimiser(self):
        parameters = {}
        choix_kernel = input(
            "Souhaiteriez-vous utiliser la variable 'kernel(noyau)' comme paramètre à optimiser ?\n0 : Non\n1 : Oui\n(defaut = Oui)\n")
        try:
            choix_kernel = int(choix_kernel)
        except:
            choix_kernel = 1
        if choix_kernel:
            parameters["kernel"] = ["linear", "rbf", "poly", "sigmoid"]

        C = input(
            "spécifier les valeurs du paramètre cout (C) : (séparer les valeurs par une virgule ex. 1,4,5,8)\nPour sélectionner plusieurs, utiliser 'debut:fin:pas' (ex. 1:20:0.1)")

        C = get_range(C)
        parameters["C"] = C

        choix_multi_class = input(
            "Souhaiteriez-vous utiliser la variable 'multi_class' comme paramètre à optimiser ?\n0 : Non\n1 : Oui\n(defaut = Oui)\n")
        try:
            choix_multi_class = int(choix_multi_class)
        except:
            choix_multi_class = 1
        if choix_multi_class:
            parameters["decision_function_shape"] = ["ovr", "crammer_singer"]

        return parameters


class SVMClassifier(Classifier, SVM):
    def __init__(self, kernel=RBF_KERNEL, cost=1, multi_class=ONE_VS_REST):
        Classifier.__init__(self, "SVM - Classifier")
        SVM.__init__(self, kernel=kernel, cost=cost, multi_class=multi_class)

    def create_model(self):
        self.modele = SVC(kernel=self.kernel, C=self.cost, decision_function_shape=self.multi_class)
        self.model = self.modele

    def afficher_parametres(self):
        Learner.afficher_parametres(self)
        print("Fonction noyau : ", self.kernel)
        print("Fonction cout :", self.cost)
        print("Méthode pour multiclasse :", self.multi_class)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()


class SVMRegressor(Regressor, SVM):
    def __init__(self, kernel=RBF_KERNEL, cost=1, multi_class=ONE_VS_REST):
        Regressor.__init__(self, "SVM - Regressor")
        SVM.__init__(self, kernel=kernel, cost=cost, multi_class=multi_class)

    def create_model(self):
        self.modele = SVR(kernel=self.kernel, C=self.cost)
        self.model = self.modele

    def afficher_parametres(self):
        Learner.afficher_parametres(self)
        print("Fonction noyau : ", self.kernel)
        print("Fonction cout :", self.cost)
        print("Méthode pour multiclasse :", self.multi_class)

    def choisir_parametres_a_optimiser(self):
        return self._choisir_parametres_a_optimiser()
