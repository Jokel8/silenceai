#!/usr/bin/env python3
"""
This program converts all-caps gloss-style German input (e.g., "GESTERN ICH GEHEN ARZT")
into a well-formed, grammatically correct German sentence. It is entirely rule-based.
"""
import sys
import argparse

class Lexicon:
    def __init__(self):
        # Pronouns: nominative pronouns with person/number and reflexive forms
        self.pronouns = {
            "ICH":   {"person":1, "number":"sg", "case":"nom", "lower":"ich", "reflexive":"mich"},
            "DU":    {"person":2, "number":"sg", "case":"nom", "lower":"du", "reflexive":"dich"},
            "ER":    {"person":3, "number":"sg", "gender":"m", "case":"nom", "lower":"er", "reflexive":"sich"},
            "SIE":   {"person":3, "number":"sg", "gender":"f", "case":"nom", "lower":"sie", "reflexive":"sich"},  # singular she (if context)
            "ES":    {"person":3, "number":"sg", "gender":"n", "case":"nom", "lower":"es", "reflexive":"sich"},
            "WIR":   {"person":1, "number":"pl", "case":"nom", "lower":"wir", "reflexive":"uns"},
            "IHR":   {"person":2, "number":"pl", "case":"nom", "lower":"ihr", "reflexive":"euch"},
            # "SIE" can also be plural or formal; our handling will treat uppercase SIE in context
            "SIE_PL": {"person":3, "number":"pl", "case":"nom", "lower":"sie", "reflexive":"sich"}
        }
        # Modal verbs with conjugation (present and simple past)
        self.modals = {
            "wollen": {"present": {"1sg":"will",   "2sg":"willst",  "3sg":"will",   "1pl":"wollen", "2pl":"wollt",  "3pl":"wollen"},
                       "past":    {"1sg":"wollte", "2sg":"wolltest","3sg":"wollte", "1pl":"wollten","2pl":"wolltet","3pl":"wollten"}},
            "müssen": {"present": {"1sg":"muss",   "2sg":"musst",   "3sg":"muss",   "1pl":"müssen", "2pl":"müsst",  "3pl":"müssen"},
                       "past":    {"1sg":"musste","2sg":"musstest","3sg":"musste", "1pl":"mussten","2pl":"musstet","3pl":"mussten"}},
            "sollen": {"present": {"1sg":"soll",   "2sg":"sollst",  "3sg":"soll",   "1pl":"sollen", "2pl":"sollt",  "3pl":"sollen"},
                       "past":    {"1sg":"sollte","2sg":"solltest","3sg":"sollte", "1pl":"sollten","2pl":"solltet","3pl":"sollten"}},
            "dürfen": {"present": {"1sg":"darf",   "2sg":"darfst",  "3sg":"darf",   "1pl":"dürfen", "2pl":"dürft",  "3pl":"dürfen"},
                       "past":    {"1sg":"durfte","2sg":"durftest","3sg":"durfte", "1pl":"durften","2pl":"durftet","3pl":"durften"}},
            "können": {"present": {"1sg":"kann",   "2sg":"kannst",  "3sg":"kann",   "1pl":"können", "2pl":"könnt",  "3pl":"können"},
                       "past":    {"1sg":"konnte","2sg":"konntest","3sg":"konnte", "1pl":"konnten","2pl":"konntet","3pl":"konnten"}},
            "mögen":  {"present": {"1sg":"mag",    "2sg":"magst",   "3sg":"mag",    "1pl":"mögen",  "2pl":"mögt",   "3pl":"mögen"},
                       "past":    {"1sg":"mochte","2sg":"mochtest","3sg":"mochte", "1pl":"mochten","2pl":"mochtet","3pl":"mochten"}}
        }
        # Common verbs lexicon with present conjugations, participles, and auxiliary
        self.verbs = {
            # Auxiliaries
            "sein":   {"present": {"1sg":"bin","2sg":"bist","3sg":"ist","1pl":"sind","2pl":"seid","3pl":"sind"},
                       "past":    {"1sg":"war","2sg":"warst","3sg":"war","1pl":"waren","2pl":"wart","3pl":"waren"},
                       "participle":"gewesen", "aux": "sein"},
            "haben":  {"present": {"1sg":"habe","2sg":"hast","3sg":"hat","1pl":"haben","2pl":"habt","3pl":"haben"},
                       "past":    {"1sg":"hatte","2sg":"hattest","3sg":"hatte","1pl":"hatten","2pl":"hattet","3pl":"hatten"},
                       "participle":"gehabt", "aux": "haben"},
            # Common verbs
            "gehen":  {"present": {"1sg":"gehe","2sg":"gehst","3sg":"geht","1pl":"gehen","2pl":"geht","3pl":"gehen"},
                       "participle":"gegangen", "aux": "sein"},
            "kommen": {"present": {"1sg":"komme","2sg":"kommst","3sg":"kommt","1pl":"kommen","2pl":"kommt","3pl":"kommen"},
                       "participle":"gekommen", "aux": "sein"},
            "essen":  {"present": {"1sg":"esse","2sg":"isst","3sg":"isst","1pl":"essen","2pl":"esst","3pl":"essen"},
                       "participle":"gegessen", "aux": "haben"},
            "trinken":{"present": {"1sg":"trinke","2sg":"trinkst","3sg":"trinkt","1pl":"trinken","2pl":"trinkt","3pl":"trinken"},
                       "participle":"getrunken", "aux": "haben"},
            "schlafen":{"present":{"1sg":"schlafe","2sg":"schläfst","3sg":"schläft","1pl":"schlafen","2pl":"schlaft","3pl":"schlafen"},
                        "participle":"geschlafen", "aux":"haben"},
            "laufen": {"present": {"1sg":"laufe","2sg":"läufst","3sg":"läuft","1pl":"laufen","2pl":"lauft","3pl":"laufen"},
                       "participle":"gelaufen", "aux":"sein"},
            "sehen":  {"present": {"1sg":"sehe","2sg":"siehst","3sg":"sieht","1pl":"sehen","2pl":"seht","3pl":"sehen"},
                       "participle":"gesehen", "aux":"haben"},
            "lesen":  {"present": {"1sg":"lese","2sg":"liest","3sg":"liest","1pl":"lesen","2pl":"lest","3pl":"lesen"},
                       "participle":"gelesen", "aux":"haben"},
            "fahren": {"present": {"1sg":"fahre","2sg":"fährst","3sg":"fährt","1pl":"fahren","2pl":"fahrt","3pl":"fahren"},
                       "participle":"gefahren", "aux":"sein"},
            "geben":  {"present": {"1sg":"gebe","2sg":"gibst","3sg":"gibt","1pl":"geben","2pl":"gebt","3pl":"geben"},
                       "participle":"gegeben", "aux":"haben"},
            "nehmen": {"present": {"1sg":"nehme","2sg":"nimmst","3sg":"nimmt","1pl":"nehmen","2pl":"nehmt","3pl":"nehmen"},
                       "participle":"genommen", "aux":"haben"},
            "sprechen":{"present":{"1sg":"spreche","2sg":"sprichst","3sg":"spricht","1pl":"sprechen","2pl":"sprecht","3pl":"sprechen"},
                        "participle":"gesprochen", "aux":"haben"},
            "sagen":  {"present": {"1sg":"sage","2sg":"sagst","3sg":"sagt","1pl":"sagen","2pl":"sagt","3pl":"sagen"},
                       "participle":"gesagt", "aux":"haben"},
            "machen": {"present": {"1sg":"mache","2sg":"machst","3sg":"macht","1pl":"machen","2pl":"macht","3pl":"machen"},
                       "participle":"gemacht", "aux":"haben"},
            "wissen": {"present": {"1sg":"weiß","2sg":"weißt","3sg":"weiß","1pl":"wissen","2pl":"wisst","3pl":"wissen"},
                       "participle":"gewusst", "aux":"haben"},
            "finden": {"present": {"1sg":"finde","2sg":"findest","3sg":"findet","1pl":"finden","2pl":"findet","3pl":"finden"},
                       "participle":"gefunden", "aux":"haben"},
            "bleiben":{"present": {"1sg":"bleibe","2sg":"bleibst","3sg":"bleibt","1pl":"bleiben","2pl":"bleibt","3pl":"bleiben"},
                       "participle":"geblieben", "aux":"sein"},
            "helfen": {"present": {"1sg":"helfe","2sg":"hilfst","3sg":"hilft","1pl":"helfen","2pl":"helft","3pl":"helfen"},
                       "participle":"geholfen", "aux":"haben"},
            "denken": {"present": {"1sg":"denke","2sg":"denkst","3sg":"denkt","1pl":"denken","2pl":"denkt","3pl":"denken"},
                       "participle":"gedacht", "aux":"haben"},
            "treffen":{"present": {"1sg":"treffe","2sg":"triffst","3sg":"trifft","1pl":"treffen","2pl":"trefft","3pl":"treffen"},
                       "participle":"getroffen", "aux":"haben"},
            "brauchen":{"present":{"1sg":"brauche","2sg":"brauchst","3sg":"braucht","1pl":"brauchen","2pl":"braucht","3pl":"brauchen"},
                        "participle":"gebraucht", "aux":"haben"},
            "spielen":{"present":{"1sg":"spiele","2sg":"spielst","3sg":"spielt","1pl":"spielen","2pl":"spielt","3pl":"spielen"},
                       "participle":"gespielt", "aux":"haben"},
            "arbeiten":{"present":{"1sg":"arbeite","2sg":"arbeitest","3sg":"arbeitet","1pl":"arbeiten","2pl":"arbeitet","3pl":"arbeiten"},
                        "participle":"gearbeitet", "aux":"haben"},
            "schreiben":{"present":{"1sg":"schreibe","2sg":"schreibst","3sg":"schreibt","1pl":"schreiben","2pl":"schreibt","3pl":"schreiben"},
                         "participle":"geschrieben", "aux":"haben"},
            "fragen":  {"present":{"1sg":"frage","2sg":"fragst","3sg":"fragt","1pl":"fragen","2pl":"fragt","3pl":"fragen"},
                        "participle":"gefragt", "aux":"haben"},
            "antworten":{"present":{"1sg":"antworte","2sg":"antwortest","3sg":"antwortet","1pl":"antworten","2pl":"antwortet","3pl":"antworten"},
                         "participle":"geantwortet", "aux":"haben"},
            "liegen":  {"present":{"1sg":"liege","2sg":"liegst","3sg":"liegt","1pl":"liegen","2pl":"liegt","3pl":"liegen"},
                        "participle":"gelegen", "aux":"haben"},
            "stehen":  {"present":{"1sg":"stehe","2sg":"stehst","3sg":"steht","1pl":"stehen","2pl":"steht","3pl":"stehen"},
                        "participle":"gestanden", "aux":"haben"},
            "werden":  {"present":{"1sg":"werde","2sg":"wirst","3sg":"wird","1pl":"werden","2pl":"werdet","3pl":"werden"},
                        "participle":"geworden", "aux":"sein"}
        }
        # Mark some verbs as reflexive (requiring a reflexive pronoun)
        reflexive_verbs = ["hinlegen", "anziehen", "ausziehen", "umziehen", "waschen", "setzen", "freuen", "beeilen"]
        for rv in reflexive_verbs:
            if rv in self.verbs:
                self.verbs[rv]["reflexive"] = True
            else:
                self.verbs[rv] = {"present": {}, "participle": "", "aux": "haben", "reflexive": True}
        # Provide present & participle for added reflexives if not already present:
        self.verbs.update({
            "hinlegen": {"present":{"1sg":"lege","2sg":"legst","3sg":"legt","1pl":"legen","2pl":"legt","3pl":"legen"},
                         "participle":"hingelegt", "aux":"haben", "reflexive":True},
            "anziehen": {"present":{"1sg":"ziehe","2sg":"ziehst","3sg":"zieht","1pl":"ziehen","2pl":"zieht","3pl":"ziehen"},
                         "participle":"angezogen", "aux":"haben", "reflexive":True},
            "ausziehen": {"present":{"1sg":"ziehe","2sg":"ziehst","3sg":"zieht","1pl":"ziehen","2pl":"zieht","3pl":"ziehen"},
                          "participle":"ausgezogen", "aux":"haben", "reflexive":True},
            "umziehen": {"present":{"1sg":"ziehe","2sg":"ziehst","3sg":"zieht","1pl":"ziehen","2pl":"zieht","3pl":"ziehen"},
                         "participle":"umgezogen", "aux":"sein", "reflexive":True},  # "umziehen" (move house) uses sein
            "waschen": {"present":{"1sg":"wasche","2sg":"wäschst","3sg":"wäscht","1pl":"waschen","2pl":"wascht","3pl":"waschen"},
                        "participle":"gewaschen", "aux":"haben", "reflexive":True},
            "setzen": {"present":{"1sg":"setze","2sg":"setzt","3sg":"setzt","1pl":"setzen","2pl":"setzt","3pl":"setzen"},
                       "participle":"gesetzt", "aux":"haben", "reflexive":True},
            "freuen": {"present":{"1sg":"freue","2sg":"freust","3sg":"freut","1pl":"freuen","2pl":"freut","3pl":"freuen"},
                       "participle":"gefreut", "aux":"haben", "reflexive":True},
            "beeilen":{"present":{"1sg":"beeile","2sg":"beeilst","3sg":"beeilt","1pl":"beeilen","2pl":"beeilt","3pl":"beeilen"},
                       "participle":"beeilt", "aux":"haben", "reflexive":True}
        })
        # Nouns with gender and plural (for articles and capitalization)
        self.nouns = {
            # People
            "MANN":    {"gender":"m", "singular":"Mann", "plural":"Männer"},
            "FRAU":    {"gender":"f", "singular":"Frau", "plural":"Frauen"},
            "KIND":    {"gender":"n", "singular":"Kind", "plural":"Kinder"},
            "FREUND":  {"gender":"m", "singular":"Freund", "plural":"Freunde"},
            "FREUNDIN":{"gender":"f", "singular":"Freundin", "plural":"Freundinnen"},
            "LEHRER":  {"gender":"m", "singular":"Lehrer", "plural":"Lehrer"},
            "LEHRERIN":{"gender":"f", "singular":"Lehrerin", "plural":"Lehrerinnen"},
            "MUTTER":  {"gender":"f", "singular":"Mutter", "plural":"Mütter"},
            "VATER":   {"gender":"m", "singular":"Vater", "plural":"Väter"},
            "OPA":     {"gender":"m", "singular":"Opa", "plural":"Opas"},
            "OMA":     {"gender":"f", "singular":"Oma", "plural":"Omas"},
            # Places
            "ARZT":    {"gender":"m", "singular":"Arzt", "plural":"Ärzte", "prep":"zu"},   # "zum Arzt"
            "ÄRZTIN":  {"gender":"f", "singular":"Ärztin", "plural":"Ärztinnen", "prep":"zu"},
            "SCHULE":  {"gender":"f", "singular":"Schule", "plural":"Schulen", "prep":"zu"}, # "zur Schule"
            "MARKT":   {"gender":"m", "singular":"Markt", "plural":"Märkte", "prep":"zu"},   # "zum Markt"
            "BAHNHOF": {"gender":"m", "singular":"Bahnhof", "plural":"Bahnhöfe", "prep":"zu"}, # "zum Bahnhof"
            "HAUS":    {"gender":"n", "singular":"Haus", "plural":"Häuser"}, # use "nach Hause"
            "KINO":    {"gender":"n", "singular":"Kino", "plural":"Kinos", "prep":"in"},  # "ins Kino"
            "PARK":    {"gender":"m", "singular":"Park", "plural":"Parks", "prep":"in"},   # "in den Park"
            "BÄCKER":  {"gender":"m", "singular":"Bäcker", "plural":"Bäcker", "prep":"zu"}, # "zum Bäcker"
            "SUPERMARKT":{"gender":"m","singular":"Supermarkt","plural":"Supermärkte", "prep":"zu"}, # "zum Supermarkt"
            "APOTHEKE":{"gender":"f", "singular":"Apotheke","plural":"Apotheken", "prep":"zu"}, # "zur Apotheke"
            "BANK":    {"gender":"f", "singular":"Bank", "plural":"Banken", "prep":"zu"},   # "zur Bank"
            "POST":    {"gender":"f", "singular":"Post", "plural":"Posten", "prep":"zu"},   # "zur Post"
            "UNIVERSITÄT":{"gender":"f","singular":"Universität","plural":"Universitäten", "prep":"zu"}, # "zur Universität"
            # Objects/Things
            "GELD":    {"gender":"n", "singular":"Geld", "plural":"Gelder"},
            "ZEIT":    {"gender":"f", "singular":"Zeit", "plural":"Zeiten"},
            "APFEL":   {"gender":"m", "singular":"Apfel", "plural":"Äpfel"},
            "BROT":    {"gender":"n", "singular":"Brot", "plural":"Brote"},
            "WASSER":  {"gender":"n", "singular":"Wasser", "plural":"Wässer"},
            "KAFFEE":  {"gender":"m", "singular":"Kaffee", "plural":"Kaffees"},
            "TEE":     {"gender":"m", "singular":"Tee", "plural":"Tees"},
            "AUTO":    {"gender":"n", "singular":"Auto", "plural":"Autos"},
            "FAHRRAD": {"gender":"n", "singular":"Fahrrad", "plural":"Fahrräder"},
            "BUCH":    {"gender":"n", "singular":"Buch", "plural":"Bücher"},
            "TISCH":   {"gender":"m", "singular":"Tisch", "plural":"Tische"},
            "STUHL":   {"gender":"m", "singular":"Stuhl", "plural":"Stühle"},
            "KATZE":   {"gender":"f", "singular":"Katze", "plural":"Katzen"},
            "HUND":    {"gender":"m", "singular":"Hund", "plural":"Hunde"},
            "VOGEL":   {"gender":"m", "singular":"Vogel", "plural":"Vögel"},
            "SCHWEIN": {"gender":"n", "singular":"Schwein", "plural":"Schweine"},
            "JUNGE":   {"gender":"m", "singular":"Junge", "plural":"Jungen"},  # note: n-noun (acc = Jungen)
            "MÄDCHEN":{"gender":"n", "singular":"Mädchen", "plural":"Mädchen"}
        }
        # Adverbs (especially time)
        self.adverbs = {
            "GESTERN": {"meaning":"yesterday", "time":"past"},
            "HEUTE":   {"meaning":"today", "time":"present"},
            "MORGEN":  {"meaning":"tomorrow", "time":"future"},
            "JETZT":   {"meaning":"now", "time":"present"},
            "SPÄTER":  {"meaning":"later", "time":"future"},
            "VORHIN":  {"meaning":"earlier", "time":"past"}
        }
        # Negation words
        self.negations = {
            "NICHT": "nicht",
            "KEIN":  "kein"
        }
        # Common misspellings mapping
        self.misspellings = {
            "ARTZT": "ARZT"  # obsolete/incorrect spelling -> correct
            # add more as needed
        }

    def is_verb_form(self, token):
        """If token is a known conjugated form of any verb or modal, return its base infinitive; otherwise None."""
        tkn = token.lower()
        # Check in verbs
        for base, data in self.verbs.items():
            # check present forms
            for form in data.get("present", {}).values():
                if tkn == form:
                    return base
            # check past forms if available
            for form in data.get("past", {}).values():
                if tkn == form:
                    return base
            # check participle
            if tkn == data.get("participle"):
                return base
        # Check modals
        for base, data in self.modals.items():
            for form in data.get("present", {}).values():
                if tkn == form:
                    return base
            for form in data.get("past", {}).values():
                if tkn == form:
                    return base
        return None

    def get_noun_form(self, noun_key, number="sg"):
        """Return the properly capitalized noun form (singular or plural)."""
        key = noun_key.upper()
        if key in self.nouns:
            if number == "pl":
                return self.nouns[key]["plural"]
            else:
                return self.nouns[key]["singular"]
        else:
            return noun_key.capitalize()

lexicon = Lexicon()

# Rule base class
class Rule:
    def __init__(self, name):
        self.name = name
    def apply(self, tokens):
        return tokens

class SpellNormalizationRule(Rule):
    def __init__(self):
        super().__init__("SpellNormalization")
    def apply(self, tokens):
        # Correct common misspellings
        return [lexicon.misspellings.get(token.upper(), token) for token in tokens]

class LemmatizationRule(Rule):
    def __init__(self):
        super().__init__("Lemmatization")
    def apply(self, tokens):
        new_tokens = []
        for token in tokens:
            base = lexicon.is_verb_form(token)
            if base:
                new_tokens.append(base)  # replace conjugated verb form with base infinitive
            else:
                new_tokens.append(token.lower())  # use lowercase for uniformity (except nouns handled later)
        return new_tokens

class NegationRule(Rule):
    def __init__(self):
        super().__init__("NegationHandling")
    def apply(self, tokens):
        # Handle "kein" inflection
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.upper() == "KEIN":
                # determine correct form of kein based on next word if it's a noun
                if i+1 < len(tokens):
                    noun_tok = tokens[i+1]
                    if noun_tok.upper() in lexicon.nouns:
                        gender = lexicon.nouns[noun_tok.upper()]["gender"]
                        # assume accusative case for objects:
                        if gender == "m":
                            new_tokens.append("keinen")
                        elif gender == "f":
                            new_tokens.append("keine")
                        elif gender == "n":
                            new_tokens.append("kein")
                        else:
                            # plural or unknown, use "keine"
                            new_tokens.append("keine")
                    else:
                        new_tokens.append("kein")
                else:
                    new_tokens.append("kein")
                i += 1
            else:
                new_tokens.append(token)
                i += 1
        # (Placement of "nicht" will be handled later in word order or conjugation stage)
        return new_tokens

class PrepositionRule(Rule):
    def __init__(self):
        super().__init__("PrepositionHandling")
    def apply(self, tokens):
        # Insert appropriate prepositions (zu, in, nach) for place nouns without a preceding preposition
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            upper_token = token.upper()
            if upper_token in lexicon.nouns and "prep" in lexicon.nouns[upper_token]:
                prep = lexicon.nouns[upper_token]["prep"]
                if prep == "zu":
                    # zu + dem/der
                    gender = lexicon.nouns[upper_token]["gender"]
                    if gender in ["m", "n"]:
                        new_tokens.append("zum")
                    elif gender == "f":
                        new_tokens.append("zur")
                    else:
                        new_tokens.append("zu")
                        if gender == "p":  # plural
                            new_tokens.append("den")
                    new_tokens.append(lexicon.get_noun_form(upper_token))
                elif prep == "in":
                    # in + dem/der (or ins)
                    gender = lexicon.nouns[upper_token]["gender"]
                    if gender == "m":
                        new_tokens.append("in")
                        new_tokens.append("den")
                    elif gender == "n":
                        new_tokens.append("ins")  # in das
                    else:  # f or plural
                        new_tokens.append("in")
                        new_tokens.append("die")
                    new_tokens.append(lexicon.get_noun_form(upper_token))
                else:
                    # e.g. special "nach"
                    if upper_token == "HAUS":
                        new_tokens.append("nach")
                        new_tokens.append("Hause")
                    else:
                        new_tokens.append(token)
                i += 1
            else:
                # If it's a noun without a 'prep' specified, or any other token, just pass it through.
                if upper_token in lexicon.nouns:
                    new_tokens.append(lexicon.get_noun_form(upper_token))
                else:
                    new_tokens.append(token)
                i += 1
        return new_tokens

class ReflexiveRule(Rule):
    def __init__(self):
        super().__init__("ReflexiveHandling")
    def apply(self, tokens):
        # If a reflexive verb is present, insert the corresponding reflexive pronoun after the subject.
        # Find if any verb in tokens requires reflexive:
        requires_reflexive = None
        for tok in tokens:
            if tok in lexicon.verbs and lexicon.verbs[tok].get("reflexive"):
                requires_reflexive = tok
                break
        if not requires_reflexive:
            return tokens
        # Identify subject pronoun and its reflexive form
        subj_index = None
        subj_refl = None
        for idx, tok in enumerate(tokens):
            if tok.upper() in lexicon.pronouns and lexicon.pronouns[tok.upper()]["case"] == "nom":
                subj_index = idx
                subj_refl = lexicon.pronouns[tok.upper()]["reflexive"]
                break
        if subj_index is not None and subj_refl:
            # Insert reflexive pronoun right after the subject (temporarily)
            if subj_refl not in [t.lower() for t in tokens]:
                tokens = tokens[:subj_index+1] + [subj_refl] + tokens[subj_index+1:]
        return tokens

class TenseRule(Rule):
    def __init__(self):
        super().__init__("TenseAdjustment")
        self.use_perfekt = False
        self.use_modal_past = False
    def apply(self, tokens):
        # Decide tense adjustments: Perfekt for past indicators, etc.
        past_indicator = any(tok in ["gestern", "vorhin", "neulich"] for tok in tokens)
        modal_present = None
        main_verb = None
        for tok in tokens:
            if tok in lexicon.modals:
                modal_present = tok
            if tok in lexicon.verbs:
                main_verb = tok
        if past_indicator:
            if modal_present:
                # If there's a modal and a past time, use Präteritum for the modal
                self.use_modal_past = True
            else:
                # Use Perfekt if past and no modal
                self.use_perfekt = True
        return tokens

class WordOrderRule(Rule):
    def __init__(self, is_question_flag=False, tense_rule=None):
        super().__init__("WordOrder")
        self.is_question_input = is_question_flag
        self.tense_rule = tense_rule
    def apply(self, tokens):
        # Determine if we need to invert for question or fronted element
        is_question = self.is_question_input
        if not is_question and tokens:
            # If first token is a verb (base form after lemma) and no explicit question mark flag, assume it was intended as question.
            if tokens[0] in lexicon.verbs or tokens[0] in lexicon.modals:
                is_question = True
        if is_question:
            # Yes/No question: verb-first
            subj_idx = None
            verb_idx = None
            for idx, tok in enumerate(tokens):
                if tok.upper() in lexicon.pronouns and subj_idx is None:
                    subj_idx = idx
                if lexicon.is_verb_form(tok) and verb_idx is None:
                    verb_idx = idx
            if verb_idx is not None:
                verb_token = tokens.pop(verb_idx)
                tokens.insert(0, verb_token)
        else:
            # Declarative: ensure V2
            if tokens:
                first_tok = tokens[0]
                if first_tok in [adv.lower() for adv in lexicon.adverbs]:
                    # Fronted element present
                    subj_idx = None
                    verb_idx = None
                    for idx, tok in enumerate(tokens):
                        if tok.upper() in lexicon.pronouns and subj_idx is None:
                            subj_idx = idx
                        if lexicon.is_verb_form(tok) and verb_idx is None:
                            verb_idx = idx
                    if self.tense_rule and getattr(self.tense_rule, 'use_perfekt', False):
                        # If Perfekt is to be used, skip swapping now (auxiliary will come in later as verb).
                        pass
                    elif subj_idx is not None and verb_idx is not None and subj_idx < verb_idx:
                        # Swap subject and verb to put verb second
                        tokens[subj_idx], tokens[verb_idx] = tokens[verb_idx], tokens[subj_idx]
        return tokens

class ConjugationRule(Rule):
    def __init__(self, tense_rule):
        super().__init__("Conjugation")
        self.tense_rule = tense_rule
    def apply(self, tokens):
        # Conjugate verbs and insert auxiliaries if needed
        # Determine subject person/number:
        person = 3; number = "sg"
        for tok in tokens:
            if tok.upper() in lexicon.pronouns:
                info = lexicon.pronouns[tok.upper()]
                person = info["person"]; number = info["number"]
                break
        # Perfekt handling:
        if getattr(self.tense_rule, 'use_perfekt', False):
            # Find main verb (last verb in tokens)
            main_idx = None
            for idx, tok in enumerate(tokens):
                if tok in lexicon.verbs:
                    main_idx = idx
            if main_idx is not None:
                base_verb = tokens[main_idx]
                aux = lexicon.verbs.get(base_verb, {}).get("aux", "haben")
                conj_aux = lexicon.verbs[aux]["present"][f"{person}{'sg' if number=='sg' else 'pl'}"]
                # Replace main verb with participle
                if base_verb in lexicon.verbs and lexicon.verbs[base_verb].get("participle"):
                    participle = lexicon.verbs[base_verb]["participle"]
                else:
                    # basic participle formation if not in lexicon
                    participle = (base_verb + "t") if base_verb.endswith("ieren") else ("ge" + base_verb + "t")
                tokens[main_idx] = participle
                # Insert auxiliary at proper position (after subject or fronted element)
                insert_pos = 0
                if tokens:
                    if tokens[0] in [adv.lower() for adv in lexicon.adverbs]:
                        insert_pos = 1
                    else:
                        insert_pos = 1
                tokens.insert(insert_pos, conj_aux)
                # Move participle to sentence end (so auxiliary and other elements come before it)
                part_idx = main_idx
                if main_idx >= insert_pos:
                    part_idx = main_idx + 1
                if part_idx < len(tokens):
                    participle_tok = tokens.pop(part_idx)
                    tokens.append(participle_tok)
        # Modal handling:
        modal_idx = None; modal_base = None
        for idx, tok in enumerate(tokens):
            if tok in lexicon.modals:
                modal_idx = idx; modal_base = tok; break
        if modal_base:
            # Conjugate modal
            tense = "present"
            if getattr(self.tense_rule, 'use_modal_past', False):
                tense = "past"
            conj_modal = lexicon.modals[modal_base][tense][f"{person}{'sg' if number=='sg' else 'pl'}"]
            tokens[modal_idx] = conj_modal
            # Ensure main verb (if any) is at end in infinitive:
            main_idx = None
            for idx, tok in enumerate(tokens):
                if idx != modal_idx and tok in lexicon.verbs:
                    main_idx = idx
            if main_idx is not None and main_idx < len(tokens) - 1:
                main_token = tokens.pop(main_idx)
                tokens.append(main_token)
        else:
            # No modal and not Perfekt: conjugate single main verb in present (or possibly Präteritum if desired)
            if not getattr(self.tense_rule, 'use_perfekt', False):
                for idx, tok in enumerate(tokens):
                    if tok in lexicon.verbs:
                        base = tok
                        if base in lexicon.verbs:
                            tokens[idx] = lexicon.verbs[base]["present"][f"{person}{'sg' if number=='sg' else 'pl'}"]
                        break
        return tokens

class CapitalizationRule(Rule):
    def __init__(self):
        super().__init__("Capitalization")
    def apply(self, tokens):
        if not tokens:
            return tokens
        tokens[0] = tokens[0].capitalize()
        # Capitalize all nouns according to lexicon
        for i, tok in enumerate(tokens[1:], start=1):
            # If token (lowercase) matches any noun's singular or plural lowercase, capitalize it
            lower_tok = tok.lower()
            for data in lexicon.nouns.values():
                if lower_tok == data["singular"].lower() or lower_tok == data.get("plural", "").lower():
                    tokens[i] = tok.capitalize()
                    break
        return tokens

def process_text(text):
    text = text.strip()
    if not text:
        return ""
    question_mark = text.endswith("?")
    # Remove punctuation for tokenization (except we noted question mark separately)
    raw = text.replace("?", "").replace("!", "").replace(".", "")
    tokens = raw.split()
    # Instantiate rule objects (with references as needed)
    tense_rule = TenseRule()
    # Determine question flag (True if input ended in '?' and no explicit question word like "WARUM")
    is_question = question_mark
    # (For W-questions, not explicitly handled in this simple implementation)
    word_order_rule = WordOrderRule(is_question_flag=is_question, tense_rule=tense_rule)
    conjugation_rule = ConjugationRule(tense_rule)
    rules = [
        SpellNormalizationRule(),
        LemmatizationRule(),
        NegationRule(),
        PrepositionRule(),
        ReflexiveRule(),
        tense_rule,
        conjugation_rule,
        word_order_rule,
        CapitalizationRule()
    ]
    for rule in rules:
        tokens = rule.apply(tokens)
    # Join tokens and add final punctuation
    sentence = " ".join(tokens)
    if is_question:
        sentence += "?"
    else:
        sentence += "."
    return sentence

def main():
    parser = argparse.ArgumentParser(description="Convert gloss-style all-caps German input to a correct German sentence.")
    parser.add_argument("--text", "-t", type=str, help="Input text in gloss form (all caps). If not provided, an interactive prompt is used.")
    args = parser.parse_args()
    if args.text:
        output = process_text(args.text)
        print(output)
    else:
        try:
            while True:
                user_input = input("Enter gloss-style German text (or 'quit' to exit): ")
                if not user_input or user_input.lower() in ("quit", "exit"):
                    break
                output = process_text(user_input)
                print(output)
        except EOFError:
            return

if __name__ == "__main__":
    main()
