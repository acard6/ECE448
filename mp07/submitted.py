'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    for rules in standardized_rules:
        if 'rule' in rules:  
          number = 'x'+rules[4:].zfill(4)
          variables.append(number)
          for val in standardized_rules[rules]:
              if val == 'antecedents':
                  for i in range(len(standardized_rules[rules][val])):
                      standardized_rules[rules][val][i] = [number if x=='something' else x for x in standardized_rules[rules][val][i]]
              if val == 'consequent':
                  standardized_rules[rules][val] = [number if x=='something' else x for x in standardized_rules[rules][val]]
      
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    qc = copy.deepcopy(query)
    dc = copy.deepcopy(datum)
    unification = None
    subs = None
    if (qc[1] == dc[1]) and (qc[3] == dc[3]):
        unification = [None, qc[1], None, qc[3]]
        subs = {}
        
        qc.pop()
        qc.pop(1)
        dc.pop()
        dc.pop(1)

        if (qc[0] not in variables):
            if dc[0] ==qc[0]:
                unification[0] = qc[0]
            else:
                subs[dc[0]] = qc[0]
                unification[0] = qc[0]

        elif (qc[0] in variables):
            subs[qc[0]] = dc[0]
            unification[0] = dc[0]

        if qc[1] == qc[0] and dc[0] != dc[1]:
            if dc[1] not in variables:
                subs[dc[0]] = dc[1]
                unification[0] = dc[1]
                unification[2] = dc[1]
            
            elif dc[0] not in variables:
                subs[dc[1]] = dc[0]
                unification[0] = dc[0]
                unification[2] = dc[0]

        elif (qc[1] in variables):
            subs[qc[1]] = dc[1]
            unification[2] = dc[1]

        elif (qc[1] not in variables):
            if dc[1] ==qc[1]:
                unification[2] = qc[1]
            else:
                subs[dc[1]] = qc[1]
                unification[2] = qc[1]
        
        if dc[1] == dc[0] and qc[0] != qc[1]:
            if qc[1] not in variables:
                unification[0] = qc[1]
                unification[2] = qc[1]
            
            elif qc[0] not in variables:
                subs[qc[1]] = qc[0]
                unification[0] = qc[0]
                unification[2] = qc[0]

        
    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    applications = []
    goalsets = []

    for g in goals:
        unification, subs = unify(rule['consequent'], g, variables)
        if subs != None:
            newRule = copy.deepcopy(rule)
            newGoal = copy.deepcopy(goals)
            
            newRule['consequent'] = unification
            if unification in newGoal:
                newGoal.remove(unification)
            for ante in newRule['antecedents']:
                while ante[0] in subs:
                    ante[0] = subs[ante[0]]
                while ante[2] in subs:
                    ante[2] = subs[ante[2]]
                newGoal.append(ante)
            
            goalsets.append(newGoal)
            applications.append(newRule)

    return applications, goalsets

def backward_chain(query, rules, variables):
    """
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    """
    proof = []
    
    def recur(rule, goal, var):
        app, newgoal = apply(rule, goal, var)
        
        if not goal.empty():
            for i in app:
                newrule = app[i]
                for j in range(len(newgoal)):
                    recur(newrule, newgoal[j], var)


    #query is my starting state
    for item in rules:
        #iterate through the rules and apply each to the query
        checkingRule = copy.deepcopy(rules[item])
        del checkingRule['text']
        #action, each rule is applied individually
        #each goalset is a state
        applications, state = apply(checkingRule, query, variables)

        if not state.empty():
            for rule in state:
                unification = unify(query, rule, variables)
        
    if proof.empty():
        return None
    return proof
