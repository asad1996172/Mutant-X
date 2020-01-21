import spacy
nlp = spacy.load('en_core_web_sm')


class Document:

    def __init__(self, inputText):

        self.documentText = ''
        self.documentWords = []
        self.documentAuthor = 0
        self.documentAuthorProbabilites = {}

        self.availableReplacements = []

        self.meteorScore = 0.0
        self.fitnessScore = 0.0
        self.classificationProbability = 0.0
        self.totalReplacements = 0.0

        doc = nlp(str(inputText))

        self.documentTrailingSpacesWords = {}
        positions = 0
        for word in doc:
            self.documentWords.append(str(word.text))
            self.documentTrailingSpacesWords[positions] = (str(word.text_with_ws), str(word.pos_))
            self.documentText+=str(word.text_with_ws)
            self.availableReplacements.append(positions)
            positions+=1
        self.wordReplacementsDict = {}

