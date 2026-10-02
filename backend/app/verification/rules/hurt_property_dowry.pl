% Voluntary hurt, theft and dowry death.

:- multifile section/2, element/4.
:- discontiguous section/2, element/4.

% s.323 Punishment for voluntarily causing hurt (s.321: causing hurt with the intention of
% causing hurt, or knowing it to be likely). The s.334 grave-provocation case is not encoded.
section('323', 'Voluntarily causing hurt').
element('323', caused_hurt, true, 'The act caused hurt to a person').
element('323', act_type, [intentional], 'The hurt was caused intentionally or knowingly (voluntarily)').

voluntary_hurt(Case) :- offence(Case, '323').

% s.379 Punishment for theft; elements of theft from s.378: "intending to take dishonestly any
% movable property out of the possession of any person without that person's consent, moves
% that property in order to such taking".
section('379', 'Theft').
element('379', movable_property, true, 'The property was movable property').
element('379', taken_from_possession, true, 'It was taken out of the possession of another person').
element('379', without_consent, true, 'It was taken without that person''s consent').
element('379', dishonest_intention, true, 'The accused intended to take it dishonestly').
element('379', property_moved, true, 'The property was moved in order to take it').

theft(Case) :- offence(Case, '379').

% s.304B Dowry death: death of a woman by burns or bodily injury or otherwise than under normal
% circumstances within seven years of marriage, where soon before her death she was subjected to
% cruelty or harassment by her husband or his relative for, or in connection with, a demand for
% dowry.
section('304B', 'Dowry death').
element('304B', woman_unnatural_death, true, 'A woman died by burns, bodily injury or otherwise than under normal circumstances').
element('304B', within_seven_years_of_marriage, true, 'The death occurred within seven years of her marriage').
element('304B', cruelty_by_husband_or_relative, true, 'She was subjected to cruelty or harassment by her husband or his relative').
element('304B', cruelty_soon_before_death, true, 'The cruelty or harassment was soon before her death').
element('304B', dowry_demand, true, 'The cruelty or harassment was for, or in connection with, a demand for dowry').

dowry_death(Case) :- offence(Case, '304B').
