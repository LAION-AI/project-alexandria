# Worked examples

Three scenes end to end: the **source text**, the **Knowledge Unit** derived from it, and the
**questions** generated from it. Chosen to show different regimes — a dialogue-heavy exchange,
a dense action scene with many entities, and a scene so short it tests the method's floor.

> **On the source text below.** This is the only file in the repository that reproduces any
> of the screenplay. Three scenes of 225, under 2% of the work, quoted for method
> illustration so a reader can judge whether the units preserve what they claim to. Every
> other artifact references the source by offset and digest and contains no source prose.
> See [provenance and scope](05-provenance-and-scope.md).

**Source.** *The Matrix* (Numbered Shooting Script, 1998-03-29), credited to Larry and Andy Wachowski. Public copy at
<https://imsdb.com/scripts/Matrix,-The.html>; sha256 `27c7163d5217805fc65317c3`.

---

## INT. HOVERCRAFT - INFIRMARY — `sc-036`

*91 words, 67% dialogue, 8 beats, 3 entities extracted.*

### 1. Source text

```text
INT. HOVERCRAFT - INFIRMARY
He opens his eyes again, something tingling through him.
He focuses and sees his body pierced with dozens of
acupuncture-like needles wired to a strange device.
DOZER
He still needs a lot of work.
DOZER and Morpheus are operating on Neo.
NEO
What are you doing?
MORPHEUS
Your muscles have atrophied.
We're rebuilding them.
Fluorescent light sticks burn unnaturally bright.
NEO
Why do my eyes hurt?
MORPHEUS
You've never used them before.
Morpheus closes Neo's eyes and Neo lays back.
MORPHEUS
Rest, Neo.  The answers are
coming.
```

### 2. Knowledge Unit

**Context before.** Neo is awake but physically weak and disoriented. He is in the infirmary of the hovercraft.

**Style.** Clinical and intimate, focusing on the physical reconstruction of Neo's body and his sensory confusion.

**Present:** `neo`, `morpheus`, `dozer`  
**Referenced:** —


**Beats** — the temporal spine. Sorting every beat in the film by
`(scene_index, order)` reconstructs the order the audience receives
information.


**1.** `[action]` `neo`  
Neo wakes up, feeling tingling sensations, and observes his body covered in needles connected to a device.  
*facts:* quantities: dozens; proper_nouns: Neo; locations: HOVERCRAFT - INFIRMARY  
*state:* `neo.consciousness`: sleeping → **awake**  
*certainty:* `stated`


**2.** `[speech]` `dozer` → `morpheus`  
Dozer notes that Neo requires significant physical repair.  
*facts:* proper_nouns: Dozer, Morpheus, Neo  
*follows from beat(s):* 1  
*certainty:* `stated`


**3.** `[speech]` `neo` → `morpheus`  
Neo asks what is being done to him.  
*facts:* proper_nouns: Neo, Morpheus  
*follows from beat(s):* 1  
*certainty:* `stated`


**4.** `[speech]` `morpheus` → `neo`  
Morpheus explains that Neo's muscles have atrophied and they are rebuilding them.  
*facts:* proper_nouns: Morpheus, Neo  
*state:* `neo.knowledge.physical_condition`: unknown → **atrophied_muscles**  
*follows from beat(s):* 3  
*certainty:* `stated`


**5.** `[speech]` `neo` → `morpheus`  
Neo asks why his eyes are painful.  
*facts:* proper_nouns: Neo, Morpheus  
*certainty:* `stated`


**6.** `[speech]` `morpheus` → `neo`  
Morpheus states that Neo has never used his eyes before.  
*facts:* proper_nouns: Morpheus, Neo  
*state:* `neo.knowledge.sensory_experience`: unknown → **first_time_vision**  
*follows from beat(s):* 5  
*certainty:* `stated`


**7.** `[action]` `morpheus` → `neo`  
Neo reclines once Morheus shuts his eyelids.  
*facts:* proper_nouns: Morpheus, Neo  
*state:* `neo.eyes`: open → **closed**  
*follows from beat(s):* 6  
*certainty:* `stated`


**8.** `[speech]` `morpheus` → `neo`  
Morpheus instructs Neo to rest, promising that answers are forthcoming.  
*facts:* proper_nouns: Morpheus, Neo  
*follows from beat(s):* 7  
*certainty:* `stated`


**Entities**

| id | name | type | attributes |
|---|---|---|---|
| `neo` | Neo | person | role: protagonist; condition: atrophied |
| `morpheus` | Morpheus | person | role: leader |
| `dozer` | Dozer | person | role: crew_member |

**Context after.** Neo is put to rest to recover, with the promise that answers will follow.


*Source reference (no text stored):* chars 39322–39879, sha256 `9506602197a65cd6`


### 3. Questions generated from this scene

Generated from the **source text**, never from the unit above.


**sc-036-q1** *(`cause`)*  
What specific physical condition of Neo's body does Morpheus cite as the reason for the procedure being performed on him?

- **A.** His muscles have atrophied ✅
- **B.** His neural pathways are damaged
- **C.** His immune system is compromised
- **D.** His skeletal structure is misaligned

**sc-036-q2** *(`cause`)*  
How does Morpheus explain to Neo why his eyes are causing him pain during the operation?

- **A.** The fluorescent lighting is too bright for his adjusted vision
- **B.** He has never used his eyes before ✅
- **C.** The needles are stimulating his optic nerves
- **D.** His corneas are drying out from the anesthesia

**sc-036-q3** *(`sequence`)*  
What specific action does Morpheus take immediately after Neo complains about his eyes hurting?

- **A.** He dims the fluorescent light sticks
- **B.** He removes the needles from Neo's face
- **C.** He closes Neo's eyes ✅
- **D.** He administers a sedative to Neo

**sc-036-q4** *(`location`)*  
Who is present in the infirmary operating on Neo alongside Morpheus?

- **A.** Trinity
- **B.** Dozer ✅
- **C.** The Oracle
- **D.** Cypher

---

## EXT. STREET — `sc-011`

*340 words, 33% dialogue, 10 beats, 10 entities extracted.*

### 1. Source text

```text
EXT. STREET
Trinity emerges from the shadows of an alley and, at the
end of the block, in a pool of white street light, she
sees it/nobreakspace--
The telephone booth.
Obviously hurt, she starts down the concrete walk,
focusing in completely, her pace quickening, as the PHONE
begins to RING.
Across the street, a garbage truck suddenly u-turns, it's
TIRES SCREAMING as it accelerates.  Trinity sees the
headlights of the truck arcing at the telephone booth as
if taking aim.
Gritting through the pain, she races the truck, slamming
into the booth, the headlights blindingly bright, bearing
down on the box of Plexiglas just as --
She answers the phone.
There is a frozen instant of silence before the hulking
mass of dark metal lurches up onto the sidewalk --
Barreling through the booth, bulldozing it into a brick
wall, SMASHING it to PLEXIGLAS PULP.
After a moment, a black loafer steps down from the cab of
the garbage truck.  Agent Smith inspects the wreckage.
There is no body.  Trinity is gone.
His jaw sets as he grinds his molars in frustration.
Agent Jones and Brown walk up behind him.
AGENT JONES
She got out.
AGENT SMITH
It doesn't matter.
AGENT BROWN
The informant is real.
Agent Smith almost smiles.
AGENT SMITH
Yes.
AGENT JONES
We have the name of their next target.
AGENT BROWN
The name is Neo.
The handset of the pay phone lays on the ground, separated
in the crash like a severed limb.
AGENT SMITH
We'll need a search running.
AGENT JONES
It's already begun.
We are SUCKED TOWARDS the mouthpiece of the phone, CLOSER
and CLOSER, until the smooth gray plastic spreads out
like a horizon and the small holes widen until we FALL
THROUGH one --
Swallowed by DARKNESS.
The DARKNESS CRACKLES with phosphorescent energy, the
word "searching" blazing in around us as we EMERGE FROM a
computer screen.
The screen flickers with windowing data as a search
engine runs with a steady relentless rhythm.
We DRIFT BACK FROM the screen and INTO --
```

### 2. Knowledge Unit

**Context before.** Trinity has escaped the building and is moving through the streets, injured. The Agents are searching for her, and a specific phone booth has been identified as her exit point.

**Style.** Tense chase sequence culminating in a narrow escape, followed by a calm, cold debriefing by the Agents, ending with a visual transition into the digital world.

**Present:** `trinity`, `agent_smith`, `agent_jones`, `agent_brown`  
**Referenced:** `neo`


**Beats** — the temporal spine. Sorting every beat in the film by
`(scene_index, order)` reconstructs the order the audience receives
information.


**1.** `[action]` `trinity`  
Trinity emerged from an alley and spotted a telephone booth illuminated by streetlights at the end of the block.  
*facts:* proper_nouns: Trinity; locations: STREET  
*certainty:* `stated`


**2.** `[action]` `trinity`  
Trinity ran toward the booth as the phone began to ring, her pace increasing despite her injuries.  
*facts:* proper_nouns: Trinity; locations: STREET  
*follows from beat(s):* 1  
*certainty:* `stated`


**3.** `[action]` `garbage_truck`  
A garbage truck made a U-turn and accelerated toward the telephone booth, its headlights aimed at the structure.  
*facts:* locations: STREET  
*certainty:* `stated`


**4.** `[action]` `trinity`  
Trinity raced the truck and entered the booth, answering the phone just as the vehicle approached.  
*facts:* proper_nouns: Trinity; locations: STREET  
*state:* `trinity.location`: street → **phone_booth**  
*follows from beat(s):* 2, 3  
*certainty:* `stated`


**5.** `[action]` `garbage_truck`  
The garbage truck mounted the sidewalk and smashed through the telephone booth, crushing it against a brick wall.  
*facts:* locations: STREET  
*state:* `phone_booth.integrity`: intact → **destroyed**  
*follows from beat(s):* 4  
*certainty:* `stated`


**6.** `[action]` `agent_smith`  
Agent Smith exited the truck and inspected the wreckage, finding no body.  
*facts:* proper_nouns: Agent Smith; locations: STREET  
*state:* `trinity.status`: present → **gone**  
*follows from beat(s):* 5  
*certainty:* `stated`


**7.** `[speech]` `agent_jones` → `agent_smith`  
Agent Jones reported that Trinity had escaped.  
*facts:* proper_nouns: Agent Jones, Trinity  
*follows from beat(s):* 6  
*certainty:* `stated`


**8.** `[speech]` `agent_brown` → `agent_smith`  
The authenticity of the source was verified by Agent Brown, who then disclosed that Neo was the designation for the upcoming objective.  
*facts:* proper_nouns: Agent Brown, Neo  
*state:* `agents.target`: trinity → **neo**  
*follows from beat(s):* 7  
*certainty:* `stated`


**9.** `[speech]` `agent_smith` → `agent_jones`  
Agent Smith requested that a search be initiated, to which Agent Jones replied that it had already begun.  
*facts:* proper_nouns: Agent Smith, Agent Jones  
*follows from beat(s):* 8  
*certainty:* `stated`


**10.** `[action]` `camera`  
The perspective moved through the phone's mouthpiece into a computer screen displaying a search engine running.  
*state:* `narrative_focus.location`: street → **computer_screen**  
*follows from beat(s):* 9  
*certainty:* `stated`


**Entities**

| id | name | type | attributes |
|---|---|---|---|
| `trinity` | Trinity | person | role: resistance_fighter; status: escaped |
| `agent_smith` | Agent Smith | person | role: agent |
| `agent_jones` | Agent Jones | person | role: agent |
| `agent_brown` | Agent Brown | person | role: agent |
| `neo` | Neo | person | role: target |
| `agents` | agents *(auto-declared)* | other | — |
| `camera` | camera *(auto-declared)* | other | — |
| `garbage_truck` | garbage truck *(auto-declared)* | other | — |
| `narrative_focus` | narrative focus *(auto-declared)* | other | — |
| `phone_booth` | phone booth *(auto-declared)* | other | — |

**Context after.** Trinity has successfully exited the Matrix via the phone booth just before it was destroyed. The Agents have confirmed her escape and identified Neo as their next target. The narrative perspective shifts to the computer screen displaying the search for Neo.


*Source reference (no text stored):* chars 7858–9810, sha256 `bec9fa232f9eebd6`


### 3. Questions generated from this scene

Generated from the **source text**, never from the unit above.


**sc-011-q1** *(`sequence`)*  
What specific action does Trinity perform immediately before the garbage truck collides with the telephone booth?

- **A.** She ducks behind a parked car to avoid the headlights.
- **B.** She answers the ringing phone. ✅
- **C.** She throws a key into the truck's cab window.
- **D.** She pulls the door shut and locks the booth.

**sc-011-q2** *(`who_did_what`)*  
Which character is the first to exit the vehicle and inspect the wreckage of the telephone booth?

- **A.** Agent Jones
- **B.** Agent Brown
- **C.** Agent Smith ✅
- **D.** The garbage truck driver

**sc-011-q3** *(`who_did_what`)*  
What piece of information do Agents Jones and Brown provide to Agent Smith regarding their next move?

- **A.** They have located Trinity's hideout.
- **B.** They have identified the informant's name as Neo. ✅
- **C.** They have secured the physical evidence from the crash site.
- **D.** They have traced the phone call to a specific area code.

**sc-011-q4** *(`location`)*  
What visual phenomenon occurs as the camera moves closer to the pay phone handset on the ground?

- **A.** The screen zooms out to reveal the entire street.
- **B.** The plastic surface expands like a horizon until the camera falls through a hole. ✅
- **C.** The phone rings again, causing the screen to flicker.
- **D.** The handset dissolves into digital code before the camera passes through it.

---

## EXT. DARK STREET — `sc-024`

*23 words, 50% dialogue, 1 beats, 1 entities extracted.*

### 1. Source text

```text
EXT. DARK STREET
A moment later the green street lights curve over the
car's tinted windshield as it rushes through the wet
underworld.
```

### 2. Knowledge Unit

**Context before.** The car has stopped and Neo has closed the door, deciding to stay inside for the procedure.

**Style.** Brief exterior shot establishing the movement of the vehicle through the rainy night.

**Present:** —  
**Referenced:** —


**Beats** — the temporal spine. Sorting every beat in the film by
`(scene_index, order)` reconstructs the order the audience receives
information.


**1.** `[action]` `vehicle`  
The car moved rapidly through a dark, wet street, with green streetlights passing over its tinted windshield.  
*facts:* locations: DARK STREET  
*state:* `vehicle.motion`: stopped → **moving**  
*certainty:* `stated`


**Entities**

| id | name | type | attributes |
|---|---|---|---|
| `vehicle` | vehicle *(auto-declared)* | other | — |

**Context after.** The vehicle is in motion again, traveling through a dark, wet urban environment.


*Source reference (no text stored):* chars 27094–27230, sha256 `889afc3eb1fbc17c`


### 3. Questions generated from this scene

Generated from the **source text**, never from the unit above.


**sc-024-q1** *(`sequence`)*  
What is the specific color of the street lighting described as curving over the car's windshield?

- **A.** Green ✅
- **B.** Amber
- **C.** Sodium-vapor orange
- **D.** Cool white

**sc-024-q2** *(`location`)*  
What specific feature of the vehicle is mentioned in relation to the street lights?

- **A.** The tinted windshield ✅
- **B.** The chrome grille
- **C.** The rearview mirror
- **D.** The open passenger window

**sc-024-q3** *(`location`)*  
What is the state of the environment described as the 'underworld' through which the car rushes?

- **A.** Wet ✅
- **B.** Foggy
- **C.** Snow-covered
- **D.** Dry and dusty

**sc-024-q4** *(`sequence`)*  
What is the implied motion of the car relative to the street lights at the start of the shot?

- **A.** Rushing through ✅
- **B.** Crawling slowly
- **C.** Stationary
- **D.** Reversing

---

## What these show

**Indirect speech.** No line of dialogue survives as dialogue. The units record what was
communicated — who told what to whom, what was refused, what was revealed — in different
words. The verbatim gate enforces this mechanically; across the whole artifact the longest
run shared with the source is 7 words.

**Facts survive exactly.** Names, numbers and locations pass through unchanged and are
listed per beat, because paraphrase applies to expression and not to fact.

**The floor is visible.** A 23-word scene yields one beat and one entity. The method does
not manufacture structure that the source does not contain, and a reader can see here
exactly how little a very short scene gives back.
