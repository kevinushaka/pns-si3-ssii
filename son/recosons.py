import glob
import pickle
from sys import argv

from pip._vendor.webencodings import labels
from python_speech_features import mfcc
import librosa
import matplotlib.pyplot as plt
#import shutil
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression

# usage: python3 recosons.py k1 k2 verbose
# ATTENTION: les noms de fichiers ne doivent comporter ni - ni espace

#sur ligne de commande: les 2 parametres de k means puis un param de verbose

k1 = int(argv[1])
k2 = int(argv[2])

if argv[3] == "True":
    verbose = True;
else:
    verbose = False;


listSons=glob.glob("Guitareélectrique/*.wav")
tmpa = len(listSons) #on mémorise le nb d'éléments de la première classe
listSons += glob.glob("GuitareAcoustique/*.wav")
#liste des labels:
groundTruth = [0]*tmpa
tmpb = len(listSons)-tmpa #nb. éléments de la snde classe
groundTruth += [1]*tmpb


lesMfcc = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
dimSons = [] # nb of mfcc per file

for s in listSons:
    if verbose:
        print("###",s,"###")
    (sig,rate) = librosa.load(s)
    mfcc_feat = mfcc(sig,rate,nfft=1024)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc,mfcc_feat,axis=0)

#BOW initialization
bows = np.empty(shape=(0,k1),dtype=int)

# everything ready for the 1st k-means
kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesMfcc)
if verbose:
    print("result of kmeans 1", kmeans1.labels_)

#writing the BOWs for second k-means
i = 0
for nb in dimSons: # for each sound (file)
    tmpBow = np.array([0]*k1)
    j = 0
    while j < nb: # for each MFCC of this sound (file)
        tmpBow[kmeans1.labels_[i]] += 1
        j+=1
        i+=1
    tmpBow = tmpBow / nb
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)
if verbose:
    print("nb of MFCC vectors per file : ", dimSons)
    print("BOWs : ", bows)
    
plt.plot(range(1,11),range(1,11),bows)
plt.show()

#ready for second k-means
kmeans2 = KMeans(n_clusters=k2, random_state=0).fit(bows)
if verbose:
    print("result of kmeans 2", kmeans2.labels_)


#écriture
with open("kmean1",'wb') as output:
    pickle.dump(kmeans1,output,pickle.HIGHEST_PROTOCOL)
with open("kmean2",'wb') as output:
    pickle.dump(kmeans2,output,pickle.HIGHEST_PROTOCOL)

#cŕeation d'un objet de regression logistique
logisticRegr = LogisticRegression()
#apprentissage
logisticRegr.fit(bows, groundTruth)
#calcul des labels pŕeditslabels
Predicted = logisticRegr.predict(bows)
#calcul et affichage du score
score = logisticRegr.score(bows, groundTruth)
print("train score = ", score)
#sauvegarde de l'objet
with open('sauvegarde.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)
#chargement de l'objet
with open('sauvegarde.logr',  'rb') as input:
    logisticRegr = pickle.load(input)






plt.scatter(lesMfcc[ : , 0], lesMfcc[ : , 1], s =50,c=kmeans1.labels_)
centroids = kmeans1.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()







listSonsToTest = glob.glob("16954_electrique.wav")
#listSonsToTest = glob.glob("17426_acoustique.wav")
#lecture
with open("kmean1","rb") as input :
    kmeans1saved = pickle.load(input)

k1 = kmeans1saved.n_clusters

if argv[3] == "True":
    verbose = True
else:
    verbose = False

#On recupère le mfcc du son
lesMfcc = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
dimSons = [] # nb of mfcc per file

for son in listSonsToTest:
    if verbose:
        print(">###", son, "###<")
    (sig, rate) = librosa.load(son)
    mfcc_feat = mfcc(sig, rate, nfft=1024)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)


#On crée le premier kmean avec les mfcc
mfccpredict = kmeans1saved.predict(lesMfcc)

#on calcule les bows
bows = np.empty(shape=(0,k1),dtype=int)

#writing the BOWs for second k-means
i = 0
for nb in dimSons: # for each sound (file)
    tmpBow = [0]*k1
    j = 0
    while j < nb: # for each MFCC of this sound (file)
        tmpBow[mfccpredict[i]] += 1
        j+=1
        i+=1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)
if verbose:
    print("nb of MFCC vectors per file : ", dimSons)
    print("BOWs :\n", bows)

#On fait la prédiction avec logisticRegr
#lecture
with open("sauvegarde.logr","rb") as input :
    logisticRegr = pickle.load(input)

prediction = logisticRegr.predict(bows)
if verbose:
    print("result of logisticRegr", prediction)

