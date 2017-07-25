
_Udacity Artificial Intelligence Nanodegree, July 2017_
# Project 3: Implement a Planning Search
The **goal** of this project is to build a planning search agent that finds the optimal shipping routes for an air cargo transport system. This is a classical planning problem. 

The project includes basic skeletons for the classes and functions needed, but students must complete the missing pieces described below.

In Part 1 of the project, we implement non-heuristics planning methods. We then run uninformed planning searches on the Air Cargo planning problems using breadth first search (BFS), depth first graph search, uniform cost search, and few other variations. 

In Part 2 of the project, we implement domain-independent heuristics to guide the search. And we analyze the results in Part 3.

Part 4 is a separate reseach paper highlighting important historic milestones in the development of AI planned search techniques. 

<p>&nbsp;</p>

---
## Part 1: Planning Problems

<p>&nbsp;</p>

#### TODO - Implement methods and functions in `my_air_cargo_problems.py`

<p>&nbsp;</p>

* __1.a__ `AirCargoProblem.get_actions` method including `load_actions` and `unload_actions` sub-functions

_Create concrete Action objects based on the domain action schema for: Load, Unload, and Fly. A concrete action is a specific literal action that does not include variables as with the schema. For example, the action schema `Load(c, p, a)` can represent the concrete actions `Load(C1, P1, SFO)` or `Load(C2, P2, JFK)`. The actions for the planning problem must be concrete because the problems in forward search and planning graphs must use propositional logic._

**Solution:** The source code for my solution can be found [here](). Below is a snippet showing the implementation of the Load action.

```python
def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        # preconditions - make sure cargo and plane are At airport
                        precond_pos = [
                            expr("At({}, {})".format(c, a)),
                            expr("At({}, {})".format(p, a)),
                        ]
                        precond_neg = []
                        # positive action - put cargo In plane
                        effect_add = [expr("In({}, {})".format(c, p))]
                        # negative action - remove cargo At airport
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        loads.append(load)
            return loads
```

* __1.b__ `AirCargoProblem.actions` method [(link to my code)]()

* __1.c__ `AirCargoProblem.result` method [(link to my code)]()

* __1.d__ `air_cargo_p2` function [(link to my code)]()

* __1.e__ `air_cargo_p3` function ([link to my code]() and snippet below)

```python
def air_cargo_p3():
    ''' Problem 3 Definition:
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD)
    	∧ At(P1, SFO) ∧ At(P2, JFK)
    	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
    	∧ Plane(P1) ∧ Plane(P2)
    	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
    Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    '''

    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['ATL', 'JFK', 'ORD', 'SFO']
    pos = [
            expr('At(C1, SFO)'),
            expr('At(C2, JFK)'),
            expr('At(C3, ATL)'),
            expr('At(C4, ORD)'),
            expr('At(P1, SFO)'),
            expr('At(P2, JFK)'),
           ]
    neg = [
            expr('At(C1, ATL)'),
            expr('At(C1, JFK)'),
            expr('At(C1, ORD)'),
            expr('In(C1, P1)'),
            expr('In(C1, P2)'),
            #
            expr('At(C2, ATL)'),
            expr('At(C2, ORD)'),
            expr('At(C2, SFO)'),
            expr('In(C2, P1)'),
            expr('In(C2, P2)'),
            #
            expr('At(C3, JFK)'),
            expr('At(C3, ORD)'),
            expr('At(C3, SFO)'),
            expr('In(C3, P1)'),
            expr('In(C3, P2)'),
            #
            expr('At(C4, ATL)'),
            expr('At(C4, JFK)'),
            expr('At(C4, SFO)'),
            expr('In(C4, P1)'),
            expr('In(C4, P2)'),
            #
            expr('At(P1, ATL)'),
            expr('At(P1, JFK)'),
            expr('At(P1, ORD)'),
            #
            expr('At(P2, ATL)'),
            expr('At(P2, ORD)'),
            expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [
            expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

```

<p>&nbsp;</p>

---
## Part 2: Domain-Independent Heuristics

<p>&nbsp;</p>

#### TODO - Implement heuristic method in `my_air_cargo_problems.py`

* __2.a__ `AirCargoProblem.h_ignore_preconditions` method [(link to my code)]()

<p>&nbsp;</p>

#### TODO - Implement a Planning Graph with automatic heuristics in `my_planning_graph.py`

* __2.b__ `PlanningGraph.add_action_level` method [(link to my code)]()

   _Add action A level to the planning graph as described in the Russell-Norvig text_

   _1. determine what actions to add and create those PgNode_a objects_

   _2. connect the nodes to the previous S literal level_

   _For example, the A0 level will iterate through all possible actions for the problem and add a PgNode_a to a_levels[0] set if all prerequisite literals for the action hold in S0.  This can be accomplished by testing to see if a proposed PgNode_a has prenodes that are a subset of the previous S level.  Once an action node is added, it MUST be connected to the S node instances in the appropriate s_level set._

__My solution:__
```python
    def add_action_level(self, level):
        """ add an A (action) level to the Planning Graph

        :param level: int
            the level number alternates S0, A0, S1, A1, S2, .... etc the level number is also used as the
            index for the node set lists self.a_levels[] and self.s_levels[]
        :return:
            adds A nodes to the current level in self.a_levels[level]
        """

        # Create empty set in level to store actions
        self.a_levels.append(set())

        # Loop through actions and determine which ones to add
        for action in self.all_actions:
            # Create an action node
            node_a = PgNode_a(action)
            # For the action node to be reachable, its preconditions must be
            # satisfied by (i.e. a subset of) the previous state level
            level_s = self.s_levels[level]
            if node_a.prenodes.issubset(level_s):
                # Connect nodes to the previous S literal level
                for node_s in level_s:
                    # Add action node as child of the S-node
                    node_s.children.add(node_a)
                    # Set S-node as the parent
                    node_a.parents.add(node_s)
                # Add A-node to current level
                self.a_levels[level].add(node_a)

```

[(link to my code for the remaining methods below)]()

* __2.c__ `PlanningGraph.add_literal_level` method
* __2.d__ `PlanningGraph.inconsistent_effects_mutex` method
* __2.e__ `PlanningGraph.interference_mutex` method
* __2.f__ `PlanningGraph.competing_needs_mutex` method
* __2.g__ `PlanningGraph.negation_mutex` method
* __2.h__ `PlanningGraph.inconsistent_support_mutex` method
* __2.i__ `PlanningGraph.h_levelsum` method

<p>&nbsp;</p>

---
## Part 3: Written Analysis
* _Provide an optimal plan for Problems 1, 2, and 3._
* _Compare and contrast non-heuristic search result metrics (optimality, time elapsed, number of node expansions) for Problems 1,2, and 3. Include breadth-first, depth-first, and at least one other uninformed non-heuristic search in your comparison; Your third choice of non-heuristic search may be skipped for Problem 3 if it takes longer than 10 minutes to run, but a note in this case should be included._
* _Compare and contrast heuristic search result metrics using A* with the "ignore preconditions" and "level-sum" heuristics for Problems 1, 2, and 3._
* _What was the best heuristic used in these problems? Was it better than non-heuristic search planning methods for all problems? Why or why not?_
* _Provide tables or other visual aids as needed for clarity in your discussion._

<p>&nbsp;</p>

### Problem 1
Below are the initial goal and state for Problem 1. This problem is relatively simple as it only involves 2 cargos, 2 airplanes, and 2 airports (JFK, SFO).

```
Init(At(C1, SFO) ∧ At(C2, JFK)
	∧ At(P1, SFO) ∧ At(P2, JFK)
	∧ Cargo(C1) ∧ Cargo(C2)
	∧ Plane(P1) ∧ Plane(P2)
	∧ Airport(JFK) ∧ Airport(SFO))
Goal(At(C1, JFK) ∧ At(C2, SFO))
```
Here are the results from all the searches that I performed, including both uninformed and heuristic searches.

![problem 1](problem-1c.jpg)

Here is the optimal plan (6 steps total). The two sets of actions are done in parallel.

```
Load(C1, P1, SFO)
Fly(P1, SFO, JFK)
Unload(C1, P1, JFK)
```
```
Load(C2, P2, JFK)
Fly(P2, JFK, SFO)
Unload(C2, P2, SFO)
```

Given the results above, the `greedy_best_first_graph_search h_1` method is the best choice. While there are several search methods which produce an optimal plan, the `greedy_best_first` does it more efficiently than the others. As highlighted during the lectures, The Greedy Best First approach works by expanding nodes closest to the goal first and therefore minimizes the number of fluents it needs to process.

The results also indicate that for simple problems like this (i.e. ones with a small search space), non-heuristic methods like `breadth_first_search` and `uniform_cost_search` are viable options. They provide reasonable efficiency and yield optimal paths without the added complexity of a heuristic. 

<p>&nbsp;</p>

### Problem 2
Below are the initial goal and state for Problem 2. This problem is slightly more complex as it now involves 3 cargos, 3 airplanes, and 3 airports (ATL, JFK, SFO).

```
Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL)
	∧ At(P1, SFO) ∧ At(P2, JFK) ∧ At(P3, ATL)
	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
	∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))
Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))
```

Here are the results from all the searches that I performed, including both uninformed and heuristic searches.

![problem 2](problem-2c.jpg)

Here is the optimal plan (9 steps total). The three sets of actions are done in parallel.

```
Load(C3, P3, ATL)
Fly(P3, ATL, SFO)
Unload(C3, P3, SFO)
```
```
Load(C2, P2, JFK)
Fly(P2, JFK, SFO)
Unload(C2, P2, SFO)
```
```
Load(C1, P1, SFO)
Fly(P1, SFO, JFK)
Unload(C1, P1, JFK)
```

Once again, we see that the `greedy_best_first` algorithm produces an optimal path with the greatest efficiency. What's more, the gain in efficiency is now much more noticable compared to `breadth_first_search` and `uniform_cost_search`. The added complexity of Problem 2 (adding 1 cargo, 1 plane, and 1 airport) makes these non-heuristic methods less viable as evidenced by longer execution time and higher number of nodes. In fact, the added complexity of this problem cause four of the other search algorithms to timeout. This demonstrates how important the size of the search space is when choosing and configuring a planning algorithm.

<p>&nbsp;</p>

### Problem 3
Below are the initial goal and state for Problem 3. This problem is the most complex since it now involves 4 cargos and 4 airports (ATL, JFK, ORD, SFO), but only 2 airplanes are available to haul everything.

```
Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD)
	∧ At(P1, SFO) ∧ At(P2, JFK)
	∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
	∧ Plane(P1) ∧ Plane(P2)
	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
```

Here are the results from all the searches that I performed, including both uninformed and heuristic searches. Although, note that some of the searches did not finish in the allotted 10-minute timeframe.


![problem 3](problem-3.jpg)

Here is the optimal plan (12 steps total). The two sets of actions are done in parallel.

```
Load(C1, P1, SFO)
Fly(P1, SFO, ATL)
Load(C3, P1, ATL)
Fly(P1, ATL, JFK)
Unload(C3, P1, JFK)
Unload(C1, P1, JFK)
```
```
Load(C2, P2, JFK)
Fly(P2, JFK, ORD)
Load(C4, P2, ORD)
Fly(P2, ORD, SFO)
Unload(C4, P2, SFO)
Unload(C2, P2, SFO)
```

<p>&nbsp;</p>

---
## Part 4: Research Review
### Instructions
The field of Artificial lIntelligence is continually changing and advancing. To be an AI Engineer at the cutting edge of your field, you need to be able to read and communicate some of these advancements with your peers. In order to help you get comfortable with this, in the second part of this project you will read a seminal paper in the field of Game-Playing and write a simple one page summary on it.

Write a simple one page summary covering the paper's goals, the techniques introduced, and results (if any).

### My Research Review
[Here is a link](https://github.com/tommytracey/udacity/tree/master/ai-nano/projects/2-isolation/results/research_review.pdf) to a PDF version of my research review on AlphaGo. The paper is titled, [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf), written by the team at Deep Mind and featured in the journal [Nature](https://www.nature.com/nature/journal/v529/n7587/full/nature16961.html) in January, 2016.

---
