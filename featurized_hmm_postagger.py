import sys
import codecs
from optparse import OptionParser
from math import log, exp
import utils
import numpy as np
from scipy.optimize import minimize
import pickle
from pprint import pprint

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

global START_SYM, END_SYM
START_SYM = "#START#"
END_SYM = "#END#"
T_TYPE = "#TRANSITION#"
E_TYPE = "#EMISSION#"
# ANY_STATE = "#ANY_STATE#"


def initialize_theta(input_weights_file, feature_index, rand=False):
    if rand:
        init_theta = np.random.uniform(1.0, 1.0, len(feature_index))
    else:
        init_theta = np.random.uniform(1.0, 1.0, len(feature_index))
    if input_weights_file is not None:
        print 'reading initial weights...'
        for l in open(input_weights_file, 'r').readlines():
            l_key = tuple(l.split()[:-1])
            if l_key in feature_index:
                init_theta[feature_index[l_key]] = float(l.split()[-1:][0])
            else:
                pass
    else:
        print 'no initial weights given, random initial weights assigned...'
    return init_theta


class FeaturizedHMMTagger(object):
    def __init__(self):
        self.possible_states = {}
        self.f_idx2f_label = {}
        self.f_lable2f_idx = {}
        self.obs2f_idx = {}
        self.corpus = []
        self.ALL_STATES = set([])
        self.fractional_counts = {}
        self.cache_normalizing_decision = {}
        self.normalizing_decision_map = {}
        self.train_method = "DGD"
        self.event2featureids = {}
        self.feature2index = {}
        self.theta = None
        self.rc = 0.001
        self.word_count = {}
        self.word_id = {}
        self.intermediate_results = None

    def initialize(self, corpus_file, feature_firing_file, possible_states_file, intermediate_results=None):
        self.intermediate_results = intermediate_results
        self.load_possible_states(possible_states_file)
        self.prepare_corpus(corpus_file)
        self.load_feature_firing(feature_firing_file)
        self.theta = np.random.uniform(1.0, 1.0, len(self.feature2index))
        sys.stderr.write('initialized\n')
        sys.stderr.write('num features:' + str(len(self.feature2index)) + '\n')
        sys.stderr.write('num event2features:' + str(len(self.event2featureids)) + '\n')


    def prepare_corpus(self, corpus_file):
        self.word_count = {}
        self.word_id = {}
        self.corpus = []
        for sent in codecs.open(corpus_file, 'r', 'utf-8').readlines():
            c = [START_SYM]
            for word in sent.split():
                c.append(word)
                for ps in self.get_possible_states(word):
                    e = (E_TYPE, word, ps)
                    f = (word, ps)
                    f_id = self._get_feature_id(f)
                    self._add_feature_to_event(e, f_id)
                self.word_id[word] = self.word_id.get(word, len(self.word_id))
                if word in self.word_count:
                    self.word_count[word] += 1
                else:
                    self.word_count[word] = 1
            c.append(END_SYM)
            self.corpus.append(c)

    def _get_feature_id(self, f):
        if f in self.feature2index:
            pass
        else:
            self.feature2index[f] = len(self.feature2index)
        return self.feature2index[f]

    def _add_feature_to_event(self, e, f_id):
        assert isinstance(e, tuple) and len(e) == 3
        assert isinstance(f_id, int)
        e2f = self.event2featureids.get(e, set([]))
        e2f.add(f_id)
        self.event2featureids[e] = e2f
        return True

    def load_feature_firing(self, feature_firing_file):
        for line in codecs.open(feature_firing_file, 'r', 'utf-8').readlines():
            items = line.split()
            obs = items[0]
            possible_states = self.get_possible_states(obs)
            for ps in possible_states:  # + [ANY_STATE]:
                e = (E_TYPE, obs, ps)
                for i in items[1:]:
                    f = (i, ps)
                    f_id = self._get_feature_id(f)
                    self._add_feature_to_event(e, f_id)

        for s1 in self.ALL_STATES + [END_SYM]:
            for s2 in self.ALL_STATES + [START_SYM]:
                if s1 == START_SYM and s2 == END_SYM:
                    pass
                else:
                    f = (s1, s2)
                    e = (T_TYPE, s1, s2)  # p(s1 | s2)
                    f_id = self._get_feature_id(f)
                    self._add_feature_to_event(e, f_id)
        self.event2featureids[E_TYPE, END_SYM, END_SYM] = []
        self.event2featureids[E_TYPE, START_SYM, START_SYM] = []


    def load_possible_states(self, possible_states_file):
        all_ps = set([])
        for line in codecs.open(possible_states_file, 'r', 'utf-8').readlines():
            ps_line = line.split()
            obs = ps_line[0]
            if obs in self.possible_states:
                sys.stderr.write("WARNING: possible states files should have unique observation tokens")
            else:
                self.possible_states[obs] = ps_line[1:]
                for ps in ps_line[1:]:
                    ndm = self.normalizing_decision_map.get((E_TYPE, ps), set([]))
                    ndm.add(obs)
                    self.normalizing_decision_map[E_TYPE, ps] = ndm

                all_ps.update(ps_line[1:])

        self.ALL_STATES = list(all_ps)

        for ps in self.ALL_STATES:
            self.normalizing_decision_map[(T_TYPE, ps)] = set([i for i in self.ALL_STATES])
        self.normalizing_decision_map[(T_TYPE, START_SYM)] = set([i for i in self.ALL_STATES])
        self.normalizing_decision_map[(E_TYPE, END_SYM)] = set([END_SYM])
        self.normalizing_decision_map[(E_TYPE, START_SYM)] = set([START_SYM])

    def get_features_fired(self, event_type, context, decision):

        ff = []
        if event_type == E_TYPE:
            ff += self.event2featureids[(E_TYPE, decision, context)]
            return ff
        else:
            e = (T_TYPE, decision, context)
            ff += self.event2featureids[e]
            return ff

    def get_decision_given_context(self, theta, type, decision, context):
        # global normalizing_decision_map, cache_normalizing_decision, feature_index, dictionary_features
        fired_features = self.get_features_fired(event_type=type,
                                                 context=context,
                                                 decision=decision)
        theta_dot_features = sum([theta[f] * 1.0 for f in fired_features])

        if (type, context) in self.cache_normalizing_decision:
            theta_dot_normalizing_features = self.cache_normalizing_decision[type, context]
        else:
            normalizing_decisions = self.normalizing_decision_map[type, context]
            theta_dot_normalizing_features = 0
            for d in normalizing_decisions:
                d_features = self.get_features_fired(event_type=type,
                                                     context=context,
                                                     decision=d)
                theta_dot_normalizing_features += exp(sum([theta[f] * 1.0 for f in d_features]))

            theta_dot_normalizing_features = log(theta_dot_normalizing_features)
            self.cache_normalizing_decision[type, context] = theta_dot_normalizing_features
        log_prob = round(theta_dot_features - theta_dot_normalizing_features, 10)
        if log_prob > 0.0:

            if self.train_method == 'DGD':
                raise Exception
            else:
                log_prob = 0.0  # TODO figure out why in the EM algorithm this error happens?
        return log_prob

    def get_possible_states(self, obs):
        if obs == END_SYM:
            return set([END_SYM])
        elif obs == START_SYM:
            return set([START_SYM])
        else:
            return self.possible_states.get(obs, self.ALL_STATES)

    def get_backwards(self, theta, obs_seq, alpha_pi, fc=None):
        n = len(obs_seq) - 1  # index of last word

        beta_pi = {(n, END_SYM): 0.0}
        S = alpha_pi[(n, END_SYM)]  # from line 13 in pseudo code
        fc = self.accumulate_fc(type=E_TYPE, alpha=0.0, beta=S, e=0.0, S=S, d=START_SYM, c=START_SYM, fc=fc)
        for k in range(n, 0, -1):
            for v in self.get_possible_states(obs_seq[k]):
                e = self.get_decision_given_context(theta=theta, type=E_TYPE, decision=obs_seq[k],
                                                    context=v)  # p(obs[k]|v)
                pb = beta_pi[(k, v)]
                fc = self.accumulate_fc(type=E_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], e=e,
                                        d=obs_seq[k], c=v, S=S, fc=fc)
                for u in self.get_possible_states(obs_seq[k - 1]):
                    q = self.get_decision_given_context(type=T_TYPE, decision=v, context=u, theta=theta)  # p(v|u)
                    fc = self.accumulate_fc(type=T_TYPE, alpha=alpha_pi[k - 1, u], beta=beta_pi[k, v],
                                            q=q, e=e, d=v, c=u, S=S, fc=fc)

                    p = q + e
                    beta_p = pb + p  # The beta includes the emission probability
                    new_pi_key = (k - 1, u)
                    if new_pi_key not in beta_pi:  # implements lines 16
                        beta_pi[new_pi_key] = beta_p
                    else:
                        beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                        alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
        if fc is None:
            return S, beta_pi
        else:
            return S, beta_pi, fc

    def get_viterbi_and_forward(self, theta, obs_seq):
        pi = {(0, START_SYM): 0.0}
        alpha_pi = {(0, START_SYM): 0.0}
        arg_pi = {(0, START_SYM): []}
        for k in range(1, len(obs_seq)):
            for v in self.get_possible_states(obs_seq[k]):  # [1]:
                max_prob_to_bt = {}
                sum_prob_to_bt = []
                for u in self.get_possible_states(obs_seq[k - 1]):  # [1]:

                    q = self.get_decision_given_context(type=T_TYPE, theta=theta, decision=v, context=u)  # p(v|u)
                    # print 'emission', k, v, obs_seq[k], (obs_seq[k],)
                    e = self.get_decision_given_context(type=E_TYPE, theta=theta, decision=obs_seq[k],
                                                        context=v)  # p(obs_seq[k] | v)

                    p = pi[(k - 1, u)] + q + e
                    alpha_p = alpha_pi[(k - 1, u)] + q + e
                    if len(arg_pi[(k - 1, u)]) == 0:
                        bt = [u]
                    else:
                        bt = [arg_pi[(k - 1, u)], u]
                    max_prob_to_bt[p] = bt
                    sum_prob_to_bt.append(alpha_p)

                max_bt = max_prob_to_bt[max(max_prob_to_bt)]
                new_pi_key = (k, v)
                pi[new_pi_key] = max(max_prob_to_bt)
                # print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
                alpha_pi[new_pi_key] = utils.logadd_of_list(sum_prob_to_bt)
                # print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
                arg_pi[new_pi_key] = max_bt

        max_bt = max_prob_to_bt[max(max_prob_to_bt)]
        max_p = max(max_prob_to_bt)
        max_bt = utils.flatten_backpointers(max_bt)
        return max_bt, max_p, alpha_pi

    def reset_fractional_counts(self):
        self.fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
        self.cache_normalizing_decision = {}

    def get_likelihood(self, theta, batch=None, display=True):
        assert isinstance(theta, np.ndarray)
        assert len(theta) == len(self.feature2index)
        if self.intermediate_results is not None:
            w = codecs.open(self.intermediate_results, 'w', 'utf-8')

        self.reset_fractional_counts()
        data_likelihood = 0.0

        if batch is None:
            batch = range(0, len(self.corpus))
        sys.stderr.write('likelihood')
        for idx in batch:
            obs_seq = self.corpus[idx]
            if idx % 1000 == 0:
                sys.stderr.write('.')
            max_bt, max_p, alpha_pi = self.get_viterbi_and_forward(theta, obs_seq)
            S, beta_pi = self.get_backwards(theta, obs_seq, alpha_pi)
            if self.intermediate_results is not None:
                w.write(' '.join(max_bt[1:]) + '\n')
            data_likelihood += S
        reg = np.sum(theta ** 2)
        ll = data_likelihood - (self.rc * reg)
        sys.stderr.write('.\n')
        if self.intermediate_results is not None:
            w.flush()
            w.close()
        if display:
            print 'log likelihood:', ll
        return -ll

    def get_gradient(self, theta, batch=None, display=True):
        assert len(theta) == len(self.feature2index)
        event_grad = {}
        sys.stderr.write('gradient')
        for idx, event_j in enumerate(self.fractional_counts):
            (t, dj, cj) = event_j
            if idx % 1000 == 0:
                sys.stderr.write('.')
            # f = self.get_features_fired(event_type=t, context=cj, decision=dj)[0]
            # TODO: resolve this why is [0]?
            a_dp_ct = exp(self.get_decision_given_context(type=t, theta=theta, decision=dj, context=cj)) * 1.0
            sum_feature_j = 0.0
            norm_events = [(t, dp, cj) for dp in self.normalizing_decision_map[t, cj]]
            for event_i in norm_events:
                A_dct = exp(self.fractional_counts.get(event_i, 0.0))
                if event_i == event_j:
                    fj = 1.0
                else:
                    fj = 0.0
                sum_feature_j += A_dct * (fj - a_dp_ct)
            if event_j not in self.event2featureids:
                sys.stderr.write('WARNING: event_j' + str(event_j) + '  is not seen in self.event2featureids\n')
            else:
                pass

            event_grad[event_j] = sum_feature_j  # - abs(theta[event_j])  # this is the regularizing term
        sys.stderr.write('.\n')
        # grad = np.zeros_like(theta)
        grad = -2 * self.rc * theta  # l2 regularization with lambda 0.5
        for e in event_grad:
            feats = self.event2featureids[e]
            for f in feats:
                grad[f] += event_grad[e]

        # grad[s] += -theta[s]  # l2 regularization with lambda 0.5
        assert len(grad) == len(self.feature2index)
        return -grad

    def accumulate_fc(self, type, alpha, beta, d, S, c=None, q=None, e=None, fc=None):
        if type == T_TYPE:
            update = alpha + q + e + beta - S
            if fc is None:
                self.fractional_counts[T_TYPE, d, c] = utils.logadd(update,
                                                                    self.fractional_counts.get(
                                                                        (T_TYPE, d, c,),
                                                                        float('-inf')))
            else:
                fc[T_TYPE, d, c] = utils.logadd(update,
                                                fc.get((T_TYPE, d, c,), float('-inf')))
        elif type == E_TYPE:
            update = alpha + beta - S  # the emission should be included in alpha
            if fc is None:
                self.fractional_counts[E_TYPE, d, c] = utils.logadd(update,
                                                                    self.fractional_counts.get(
                                                                        (E_TYPE, d, c,),
                                                                        float('-inf')))
            else:
                fc[E_TYPE, d, c] = utils.logadd(update,
                                                fc.get((E_TYPE, d, c,), float('-inf')))
        else:
            raise "Wrong type"
        if fc is not None:
            return fc


    def train(self, maxiter=20, tol=1e-3):
        self.theta = np.random.uniform(-0.1, 0.1, len(self.feature2index))
        t1 = minimize(self.get_likelihood, self.theta, method='L-BFGS-B', jac=self.get_gradient, tol=tol,
                      options={'maxiter': maxiter})
        self.theta = t1.x
        return self.theta

    def save(self, save_file):
        pickle.dump(self, file(save_file, 'w'))

    def tag(self, sentence):
        obs_seq = [START_SYM] + sentence.split() + [END_SYM]
        max_bt, max_p, alpha_pi = self.get_viterbi_and_forward(self.theta, obs_seq)
        return max_bt


if __name__ == "__main__":
    opt = OptionParser()
    # insert options here
    opt.add_option('--possible-states', dest='possible_states', default='data/possible.tags')
    opt.add_option('--features-fired', dest='features_fired', default='data/ps.features')
    opt.add_option('--corpus', dest='corpus_file', default='data/corpus.ps-en.ps')
    opt.add_option('--intermediate-results', dest='intermediate_results')
    (options, _) = opt.parse_args()
    fhmm = FeaturizedHMMTagger()
    fhmm.initialize(options.corpus_file, options.features_fired, options.possible_states, options.intermediate_results)
    # fhmm.get_likelihood(fhmm.theta)

    fhmm.train(maxiter=5)
    fhmm.save('saved.model')
    # fhmm2 = pickle.load(file('saved.model'))
    # print fhmm2.tag("The Man is eating .")

