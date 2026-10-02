% Core verification logic, shared by every encoded section.
%
% Facts about a case are asserted as  fact(CaseId, Predicate, Value)  where Value is true,
% false, unknown, or (for act_type) one of rash / negligent / intentional / accidental.
% Unknown is explicit on purpose: plain negation-as-failure (\+ fact(C, intent_to_kill, true))
% would read "the facts do not say" as "false" and wrongly satisfy a "no intention" element.
%
% Each section file declares
%   section(Section, Title).
%   element(Section, Predicate, Required, Description).
% where Required is a value or a list of acceptable values.

:- dynamic fact/3.
% Section files add clauses to these from several files.
:- multifile section/2, element/4.
:- discontiguous section/2, element/4.

satisfies(Value, Required) :- is_list(Required), !, memberchk(Value, Required).
satisfies(Value, Value).

known(Case, Pred, Value) :- fact(Case, Pred, Value), Value \== unknown.

% element_status(+Case, +Section, ?Pred, -Status): satisfied | violated | missing
element_status(Case, Section, Pred, Status) :-
    element(Section, Pred, Required, _),
    (   known(Case, Pred, Value)
    ->  ( satisfies(Value, Required) -> Status = satisfied ; Status = violated )
    ;   Status = missing
    ).

% verdict(+Case, +Section, ?Verdict): a known violation decides; otherwise any gap leaves the
% verdict open; only a complete, matching checklist is consistent.
% The verdict is computed first and only then unified with the caller's argument: with the
% "cut after the first matching clause" pattern alone, a call such as
% verdict(C, S, consistent) would skip the violated/missing clauses and wrongly succeed.
verdict(Case, Section, Verdict) :-
    section(Section, _),
    compute_verdict(Case, Section, Computed),
    Verdict = Computed.

compute_verdict(Case, Section, inconsistent) :- element_status(Case, Section, _, violated), !.
compute_verdict(Case, Section, insufficient) :- element_status(Case, Section, _, missing), !.
compute_verdict(_Case, _Section, consistent).

offence(Case, Section) :- verdict(Case, Section, consistent).
