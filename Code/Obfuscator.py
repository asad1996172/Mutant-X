import argparse
import random
import math
import gensim.models.keyedvectors as word2vec
from Document import Document
from Classifier import Classifier
from Mutant import Mutant_X
from operator import itemgetter
import copy
import os
import re
import numpy as np
import time
import io
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import writeprintsStatic

def computeNeighbours(word, wordVectorModel, neighborsCount=5):
    word = str(word).lower()
    tim = 0
    try:
        start = time.time()
        neighbors = list(wordVectorModel.similar_by_word(word, topn=neighborsCount))
        end= time.time()
        tim = end - start
    except:
        return -1,tim

    updated_neigbours = []
    for neighbor in neighbors:
        if neighbor[1] > 0.75:
            updated_neigbours.append(neighbor[0])
    if not updated_neigbours:
        return -1,tim
    # word_selected = random.choice(updated_neigbours)

    return updated_neigbours,tim

def getWordNeighboursDictionary(inputText,totalNeighbours):
    wordVectorModel = word2vec.KeyedVectors.load_word2vec_format("Word Embeddings/Word2vec.bin",
                                                                 binary=True)
    average_time = []
    word_neighbours = {}
    for word in inputText:

        neighbors,tim = computeNeighbours(word, wordVectorModel,totalNeighbours)
        if len(average_time)!=100 and tim!=0:
            average_time.append(tim)

        if neighbors != -1:
            word_neighbours[word] = neighbors
    return word_neighbours

def getRunAndDocumentNumber(indexNo, dataset_name, authorsToKeep):
    indexNo = int(indexNo)
    indexNo = indexNo - 1

    if dataset_name == 'amt' and authorsToKeep == 5:
        indexNo = indexNo % 300
    elif dataset_name == 'amt' and authorsToKeep == 10:
        indexNo = indexNo % 490
    elif dataset_name == 'BlogsAll' and authorsToKeep == 5:
        indexNo = indexNo % 1000
    elif dataset_name == 'BlogsAll' and authorsToKeep == 10:
        indexNo = indexNo % 2000

    passNo = indexNo % 10
    passNo = passNo + 1
    documentNumber = math.floor(indexNo / 10)
    documentNumber = documentNumber + 1
    random.seed()

    return passNo, documentNumber

def saveDocument(document,qualitative_results_folder, iteration_m, obfuscated=False):
    if obfuscated:
        qualitativeoutputResults = open(qualitative_results_folder + "/Obfuscated_Text", "w")
    else:
        qualitativeoutputResults = open(qualitative_results_folder + "/BestChangeInIteration_" + str(iteration_m), "w")
    qualitativeoutputResults.write(document.documentText)
    qualitativeoutputResults.write('\n')
    qualitativeoutputResults.write('\n')
    qualitativeoutputResults.write(json.dumps(document.wordReplacementsDict))
    qualitativeoutputResults.close()

def getInformationOfInputDocument(documentPath):
    authorslabeler = LabelEncoder()
    authorslabeler.classes_ = np.load('../classes.npy')

    inputText = io.open(documentPath, "r", errors="ignore").readlines()
    inputText = ''.join(str(e) + "" for e in inputText)

    authorName = (documentPath.split('/')[-1]).split('_')[0]
    authorLabel = authorslabeler.transform([authorName])[0]

    return (authorLabel, authorName, inputText)

def main():

    #################################################
    # Parameters
    #################################################
    parser = argparse.ArgumentParser()

    # Mutant-X parameters
    parser.add_argument("--generation", "-l", help="Number of documents generated per document", default=5)
    parser.add_argument("--topK", "-k", help="Top K highest fitness selection", default=5)
    parser.add_argument("--crossover", "-c", help="Crossover probability", default=0.1)
    parser.add_argument("--iterations", "-M", help="Total number of iterations", default=25)
    parser.add_argument("--alpha", "-a", help="weight assigned to probability in fitness function", default=0.75)
    parser.add_argument("--beta", "-b", help="weight assigned to METEOR in fitness function", default=0.25)
    parser.add_argument("--replacements", "-Z", help="percentage of document to change", default=0.05)
    parser.add_argument("--replacementsLimit", "-rl", help="replacements limit", default=0.20)

    # Obfuscator parameters
    parser.add_argument("--authorstoKeep", "-atk", help="Total number of authors under observation", default=5)
    parser.add_argument("--datasetName", "-dn", help="Name of dataset to test with", default='amt')
    parser.add_argument("--allowedNeighbours", "-an", help="Total allowed neighbours in word embedding", default=5)
    parser.add_argument("--documentName", "-docN", help="Name of document for obfuscation", default='h_13_2.txt')
    parser.add_argument("--classifierType", "-ctype", help="Type of classifier",default='ml')


    args = parser.parse_args()

    generation = int(args.generation)
    topK = int(args.topK)
    crossover = float(args.crossover)
    iterations = int(args.iterations)
    alpha = float(args.alpha)
    beta = float(args.beta)
    replacements = float(args.replacements)
    replacementsLimit = float(args.replacementsLimit)

    authorstoKeep = int(args.authorstoKeep)
    datasetName = args.datasetName
    documentName = args.documentName
    allowedNeighbours = int(args.allowedNeighbours)
    classifierType = args.classifierType

    runNumber = 1

    ####################################################
    # Loading Document to be Obfuscated
    ####################################################

    clf = Classifier(classifierType, authorstoKeep, datasetName, documentName)
    clf.loadClassifier()
    # testInstancesFilename = "../../Data/datasetPickles/" + str(datasetName) + '-' + str(authorstoKeep) + '/X_test.pickle'
    # with open(testInstancesFilename, 'rb') as f:
    #     testInstances = pickle.load(f)
    #
    # print("Test Instances Length : ", len(testInstances))
    filePath, filename = '../' + documentName, documentName
    authorId, author, inputText = getInformationOfInputDocument(filePath)

    print("Document Name : ", filename)

    originalDocument = Document(inputText)
    clf.getLabelAndProbabilities(originalDocument)

    if originalDocument.documentAuthor != authorId:
        print("Classified InCorrectly , Hence Skipping the Article")
        return

    if os.path.isfile('documentsWordsSpace/' + filename.split('.')[0] + '.pickle'):
        print('Word Space Dictionary exists !!')
        with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'rb') as handle:
            neighboursDictionary = pickle.load(handle)
    else:
        print('Creating Word Space Dictionary !!')
        neighboursDictionary = getWordNeighboursDictionary(originalDocument.documentWords,allowedNeighbours)
        with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'wb') as handle:
            pickle.dump(neighboursDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ####################################################
    # Setting up directories and loading Mutant-X
    ####################################################

    if not os.path.exists(classifierType + "Results" + datasetName + str(authorstoKeep)):
        os.makedirs(classifierType + "Results" + datasetName + str(authorstoKeep))

    if not os.path.exists(classifierType + "Results" + datasetName+ str(authorstoKeep) + "/" + filename):
        os.makedirs(classifierType + "Results" + datasetName + str(authorstoKeep) + "/" + filename)

    if not os.path.exists(classifierType + "QualitativeResults" + datasetName + str(authorstoKeep)):
        os.makedirs(classifierType + "QualitativeResults" + datasetName + str(authorstoKeep))

    if not os.path.exists(classifierType + "QualitativeResults" + datasetName+ str(authorstoKeep) + "/" + filename):
        os.makedirs(classifierType + "QualitativeResults" + datasetName+ str(authorstoKeep) + "/" + filename)

    mutant_x = Mutant_X(generation, topK, crossover, iterations, neighboursDictionary, alpha, beta, replacements, replacementsLimit)

    print('ARTICLE ====> ', filename, "    Author ====> ", author, "     Pass NO. ====> ", runNumber)

    runNumber = str(runNumber) + "_" + str(time.time())
    created_file_name = classifierType + "Results" + datasetName + str(authorstoKeep) + "/" + str(filename) + "/" + str(
        runNumber) + "_" + str(generation) + "_" + str(topK) + "_" \
                        + str(originalDocument.documentAuthorProbabilites[originalDocument.documentAuthor])\
                        + "_" + str(authorId) + "_" + str(
        originalDocument.documentAuthor) + ".csv"

    qualitative_results_folder = classifierType + "QualitativeResults" + datasetName + str(
        authorstoKeep) + "/" + filename + "/" + str(runNumber)

    if not os.path.exists(qualitative_results_folder):
        os.makedirs(qualitative_results_folder)

    qualitativeoutputResults = open(qualitative_results_folder + "/Orignal_Text", "w")
    qualitativeoutputResults.write(originalDocument.documentText)
    qualitativeoutputResults.close()

    prob_str = ''
    prob_values = ''
    for i in range(authorstoKeep):
        prob_str += ','
        prob_str += str(i)
        prob_values+= ','
        prob_values+=str(originalDocument.documentAuthorProbabilites[i])

    outputResults = open(created_file_name, "w")
    outputResults.write(
        'iterationNo' + "," + 'fitness' + "," + 'probOfDetection' + "," + 'numOfReplacements' + "," + 'PredictedLabel' + "," + 'meteorScore' + prob_str + "\n")
    outputResults.write("0,0," + str(originalDocument.documentAuthorProbabilites[originalDocument.documentAuthor]) + ",0," + str(originalDocument.documentAuthor) + ",0" + prob_values + "\n")
    print('STARTING PROBABILITY : ', originalDocument.documentAuthorProbabilites[originalDocument.documentAuthor])

    ####################################################
    # Starting Mutant-X Obfuscation Process
    ####################################################

    indivisualsPopulation = [originalDocument]
    iteration_m = 1
    obfuscated = False
    while (iteration_m < mutant_x.iterations) and (obfuscated == False) :

        print("Iteration =====> ", iteration_m)

        # Generation Process
        generatedPopulation = []
        for indivisual in indivisualsPopulation:
            for i in range(0, mutant_x.generation):
                indivisualCopy = copy.deepcopy(indivisual)
                genDocument = mutant_x.makeReplacement(indivisualCopy)
                clf.getLabelAndProbabilities(genDocument)
                generatedPopulation.append(genDocument)

        indivisualsPopulation.extend(generatedPopulation)

        # Crossover Process
        if random.random() < mutant_x.crossover:
            print("CROSSING OVER")
            choice1, choice2 = random.sample(indivisualsPopulation, 2)
            choice1Copy = copy.deepcopy(choice1)
            choice2Copy = copy.deepcopy(choice2)
            child_1, child_2 = mutant_x.single_point_crossover(choice1Copy, choice2Copy)

            clf.getLabelAndProbabilities(child_1)
            clf.getLabelAndProbabilities(child_2)

            indivisualsPopulation.extend([child_1, child_2])


        if originalDocument in indivisualsPopulation:
            indivisualsPopulation.remove(originalDocument)

        # Calculating Fitness

        for indivisual in indivisualsPopulation:

            mutant_x.calculateFitness(originalDocument,indivisual)

            if indivisual.documentAuthor != originalDocument.documentAuthor:
                print("Obfuscated Successfully !!!")
                obfuscated = True
                saveDocument(indivisual, qualitative_results_folder, 0,obfuscated=True)


            prob_str = ''
            for i in range(authorstoKeep):
                prob_str += ','
                prob_str += str(indivisual.documentAuthorProbabilites[i])

            outputResults.write(
                str(iteration_m) + "," + str(indivisual.fitnessScore) + "," + str(indivisual.classificationProbability) + "," + str(
                    indivisual.totalReplacements) + "," + str(indivisual.documentAuthor) + "," + str(indivisual.meteorScore) + prob_str + "\n")



        # Selecting topK

        indivisualsPopulation.sort(key=lambda x:x.fitnessScore, reverse=True)
        indivisualsPopulation = indivisualsPopulation[:mutant_x.topK]

        bestIndivisual = indivisualsPopulation[0]
        saveDocument(bestIndivisual, qualitative_results_folder, iteration_m, obfuscated=False)

        for topDocument in indivisualsPopulation:
            print('CURRENT PROBABILITY : ',
                  topDocument.documentAuthorProbabilites[originalDocument.documentAuthor])

        iteration_m+=1
    outputResults.close()


if __name__ == "__main__":
    main()