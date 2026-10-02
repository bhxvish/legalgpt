% Offences of rashness and negligence (IPC Chapters XIV and XVI).
% Element lists follow the section text in the bare act (data/raw_corpus, ingested in Phase 1).

:- multifile section/2, element/4.
:- discontiguous section/2, element/4.

% s.304A Causing death by negligence: "Whoever causes the death of any person by doing any rash
% or negligent act not amounting to culpable homicide". "Not amounting to culpable homicide" is
% encoded as the absence of both mental states of s.299 (intention to cause death / such bodily
% injury as is likely to cause death, and knowledge that the act is likely to cause death).
section('304A', 'Causing death by negligence').
element('304A', caused_death, true, 'The act caused the death of a person').
element('304A', act_type, [rash, negligent], 'The act was rash or negligent').
element('304A', intent_to_kill, false, 'No intention to cause death or bodily injury likely to cause death').
element('304A', knowledge_likely_death, false, 'No knowledge that the act was likely to cause death').

negligent_death(Case) :- offence(Case, '304A').

% s.279 Rash driving or riding on a public way: "drives any vehicle, or rides, on any public way
% in a manner so rash or negligent as to endanger human life, or to be likely to cause hurt or
% injury to any other person".
section('279', 'Rash driving or riding on a public way').
element('279', drove_vehicle, true, 'The accused drove a vehicle or rode').
element('279', on_public_way, true, 'This happened on a public way').
element('279', act_type, [rash, negligent], 'The driving or riding was rash or negligent').
element('279', endangered_safety, true, 'It endangered human life or was likely to cause hurt or injury').

rash_driving(Case) :- offence(Case, '279').

% s.337 Causing hurt by act endangering life or personal safety of others.
section('337', 'Causing hurt by act endangering life or personal safety of others').
element('337', caused_hurt, true, 'The act caused hurt to a person').
element('337', act_type, [rash, negligent], 'The act was rash or negligent').
element('337', endangered_safety, true, 'The act endangered human life or the personal safety of others').

negligent_hurt(Case) :- offence(Case, '337').

% s.338 Causing grievous hurt by act endangering life or personal safety of others.
section('338', 'Causing grievous hurt by act endangering life or personal safety of others').
element('338', caused_grievous_hurt, true, 'The act caused grievous hurt (s.320) to a person').
element('338', act_type, [rash, negligent], 'The act was rash or negligent').
element('338', endangered_safety, true, 'The act endangered human life or the personal safety of others').

negligent_grievous_hurt(Case) :- offence(Case, '338').
