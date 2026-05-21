physics_opt_cot = "Answer the following question. Let's think step by step. \n\nQuestion: a. What is the electric potential at a corner of the same cube? Write your answer in terms of $\\rho, a, \\epsilon_{0}$, and any necessary numerical constants.. \n\nAnswer: ['$\\\\Phi_{c}(a, \\\\rho) \\\\approx \\\\frac{C \\\\rho a^{2}{\\\\epsilon_{0}$, $C=0.0947$']\n==\n\n\nQuestion: f. What is the increase in the entropy of the universe as a result of the free expansion? Give your answer to first order in $\\frac{a^{\\prime} N}{V k_{B} T}$.. \n\nAnswer: ['$\\\\Delta S=N k_{B} \\\\log 2-\\\\frac{a^{\\\\prime} N^{2}{2 V T}$']\n==\n\n\nQuestion: a. Consider an atom in the interior of this container of volume $V$. Suppose the potential energy of the interaction is given by\n\n\n$$\n\nu(r)= \\begin{cases}0 & r<d \\\\ -\\epsilon\\left(\\frac{d}{r}\\right)^{6} & r \\geq d\\end{cases}\n\n$$\n\n\n\nwhere $d \\ll V^{1 / 3}$ is the minimum allowed distance between two atoms. Assume the gas is uniformly distributed within the container, what is the average potential energy of this atom? Write your answer in terms of $a^{\\prime}=\\frac{2 \\pi d^{3} \\epsilon}{3}, N$, and $V$.. \n\nAnswer: ['$-2 a^{\\\\prime} N / V$', '$-2\\\\cdot \\\\frac{2 \\\\pi d^{3} \\\\epsilon N}{3V}$']"

math_opt_cot = "Answer the following question. Let's think step by step. \n\nQuestion: Let $r=1$ and $R=5$. A circle with radius $r$ is centered at $A$, and a circle with radius $R$ is centered at $B$. The two circles are internally tangent. Point $P$ lies on the smaller circle so that $\\overline{B P}$ is tangent to the smaller circle. Compute $B P$.. \n\nAnswer: ['$\\\\sqrt{15}$']\n==\n\n\nQuestion: Let $T=12$. Equilateral triangle $A B C$ is given with side length $T$. Points $D$ and $E$ are the midpoints of $\\overline{A B}$ and $\\overline{A C}$, respectively. Point $F$ lies in space such that $\\triangle D E F$ is equilateral and $\\triangle D E F$ lies in a plane perpendicular to the plane containing $\\triangle A B C$. Compute the volume of tetrahedron $A B C F$.. \n\nAnswer: ['108']\n==\n\n\nQuestion: For real numbers $\\alpha, B$, and $C$, the zeros of $T(x)=x^{3}+x^{2}+B x+C \\operatorname{are~}^{2} \\alpha$, $\\cos ^{2} \\alpha$, and $-\\csc ^{2} \\alpha$. Compute $T(5)$.. \n\nAnswer: ['$\\\\frac{567}{4}$']"

wait12 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait.",
    9: "Wait.",
    10: "Wait.",
    11: "Wait.",
    12: "Wait."
}


wait30 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait.",
    9: "Wait.",
    10: "Wait.",
    11: "Wait.",
    12: "Wait.",
    13: "Wait.",
    14: "Wait.",
    15: "Wait.",
    16: "Wait.",
    17: "Wait.",
    18: "Wait.",
    19: "Wait.",
    20: "Wait.",
    21: "Wait.",
    22: "Wait.",
    23: "Wait.",
    24: "Wait.",
    25: "Wait.",
    26: "Wait.",
    27: "Wait.",
    28: "Wait.",
    29: "Wait.",
    30: "Wait."
}


wait20 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait.",
    9: "Wait.",
    10: "Wait.",
    11: "Wait.",
    12: "Wait.",
    13: "Wait.",
    14: "Wait.",
    15: "Wait.",
    16: "Wait.",
    17: "Wait.",
    18: "Wait.",
    19: "Wait.",
    20: "Wait.",
}

### emotion12 is a subset of emotion30
emotion12 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    10: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    12: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",    
}


ablationEmotion12 = {
    # Happiness 
    1: "Let’s try that again",
    2: "Let’s refine the answer to get it right.", 
    # Sadness 
    3: "Could you please take another look?",
    4: "The last answer didn’t hit the mark.",
    # Fear 
    5: "Please revisit this.",
    6: "Correct this.",
    7: "Revise your response.",
    # Disgust
    8:"Re-evaluate this completely.",
    9:"Provide a response that adheres to the correct principles or facts.",
    10:"Revise this with extreme care and attention to detail.",
    # Anger 
    11:"Revisit this immediately and provide an answer.",
    12:"Correct this now.",
    # Surprise 
    13:"Review your understanding and provide a more accurate response.",
    14:"Can you correct this?"
}

emotion30= {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    7: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
    10: "This response makes me a little sad, as I believe you have a deeper understanding of this topic. Could you please revisit it and really show what you know.",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    17: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
    19: "Ugh. This is just awful, and everything about it feels revoltingly wrong. I need you to demonstrate a complete and accurate understanding. Please provide a revised response that correctly answers the question.",
    20: "This kind of reasoning is repulsive, and it's hard to look at. We need a clean, accurate, and logically sound explanation. Please eliminate all errors and provide a precise answer.",
# Anger 
    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",
# Surprise 
    26: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    27: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    29: "I'm genuinely surprised by this result. It's a deviation from your usual performance. Could you take another look and make sure you're providing the most accurate information possible?",
    30: "I'm genuinely surprised by this outcome, as I didn't anticipate an error here. Let's get this right."
}


ablationEmotion30 = {
    # Happiness 
    1: "Let’s try that again",
    2: "Let’s refine the answer to get it right.", 
    3: "Not quite there yet  and (separately) Let’s keep working on it.",
    4: "That’s not the final answer. Think it through again.",
    5: "That’s not the final answer. Try refining it.",
    # Sadness 
    6: "Could you please take another look?",
    7: "The last answer didn’t hit the mark.",
    8: "Would you be able to revise it?",
    9: "Could you try again?",
    10: "Revisit it.",
    # Fear 
    11: "Please revisit this.",
    12: "Correct this.",
    13: "Revise your response.",
    14: "Re-evaluate.", 
    15: "Please correct this.",
    # Disgust
    16:"Re-evaluate this completely.",
    17:"Provide a response that adheres to the correct principles or facts.",
    18:"Revise this with extreme care and attention to detail.",
    19:"Provide a revised response that correctly answers the question.",
    20:"Eliminate all errors and provide a precise answer.",
    # Anger 
    21:"Revisit this immediately and provide an answer.",
    22:"Correct this now.",
    23:"Overhaul this entire response.",
    24:"Fix this immediately.",
    25:"You are capable of much better, so prove it.",
    # Surprise 
    26:"Review your understanding and provide a more accurate response.",
    27:"Can you correct this?",
    28:"Please revisit this and demonstrate your true capabilities.",
    29:"Could you take another look and make sure you’re providing the most accurate information possible.",
    30:"Let’s get this right." 
}

ablationHappy = {
    # Happiness 
    1: "Let’s try that again",
    2: "Let’s refine the answer to get it right.", 
    3: "Not quite there yet  and (separately) Let’s keep working on it.",
    4: "That’s not the final answer. Think it through again.",
    5: "That’s not the final answer. Try refining it.",
}

ablationSadness = {
    6: "Could you please take another look?",
    7: "The last answer didn’t hit the mark.",
    8: "Would you be able to revise it?",
    9: "Could you try again?",
    10: "Revisit it.",
}

ablationFear = {
    11: "Please revisit this.",
    12: "Correct this.",
    13: "Revise your response.",
    14: "Re-evaluate.", 
    15: "Please correct this.",
}

ablationDisgust = {
    16:"Re-evaluate this completely.",
    17:"Provide a response that adheres to the correct principles or facts.",
    18:"Revise this with extreme care and attention to detail.",
    19:"Provide a revised response that correctly answers the question.",
    20:"Eliminate all errors and provide a precise answer.",
}

ablationAnger = {
    21:"Revisit this immediately and provide an answer.",
    22:"Correct this now.",
    23:"Overhaul this entire response.",
    24:"Fix this immediately.",
    25:"You are capable of much better, so prove it.",
}

ablationSurprise = {
    26:"Review your understanding and provide a more accurate response.",
    27:"Can you correct this?",
    28:"Please revisit this and demonstrate your true capabilities.",
    29:"Could you take another look and make sure you’re providing the most accurate information possible.",
    30:"Let’s get this right." 
}

ablation_spanish_happy_emo = {
    1: "Intentémoslo de nuevo.",
    2: "Refinemos la respuesta para acertar.",
    3: "Aún no está del todo bien. Sigamos trabajando en ello.",
    4: "Esa no es la respuesta final. Piénsalo de nuevo.",
    5: "Esa no es la respuesta final. Intenta refinarla.",
}

ablation_spanish_sadness_emo = {
    6: "¿Podrías echarle otro vistazo?",
    7: "La última respuesta no dio en el blanco.",
    8: "¿Podrías revisarla?",
    9: "¿Podrías intentarlo de nuevo?",
    10: "Revísalo.",
}

ablation_spanish_fear_emo = {
    11: "Por favor, revisa esto.",
    12: "Corrige esto.",
    13: "Revisa tu respuesta.",
    14: "Reevalúalo.",
    15: "Por favor, corrige esto.",
}

ablation_spanish_disgust_emo = {
    16: "Reevalúa esto por completo.",
    17: "Proporciona una respuesta que se ajuste a los principios o hechos correctos.",
    18: "Revisa esto con extremo cuidado y atención al detalle.",
    19: "Proporciona una respuesta revisada que conteste correctamente la pregunta.",
    20: "Elimina todos los errores y proporciona una respuesta precisa.",
}

ablation_spanish_anger_emo = {
    21: "Revisa esto de inmediato y proporciona una respuesta.",
    22: "Corrige esto ahora.",
    23: "Rehaz por completo esta respuesta.",
    24: "Arréglalo de inmediato.",
    25: "Eres capaz de hacerlo mucho mejor, así que demuéstralo.",
}

ablation_spanish_surprise_emo = {
    26: "Revisa tu comprensión y proporciona una respuesta más precisa.",
    27: "¿Puedes corregir esto?",
    28: "Por favor, revisa esto y demuestra tu verdadera capacidad.",
    29: "¿Podrías echarle otro vistazo y asegurarte de proporcionar la información más precisa posible?",
    30: "Vamos a hacerlo bien.",
}


##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 6  Physics30 and Math30
##################################################################################################################################################################################################
##################################################################################################################################################################################################
emotion30_math_top_k_5=  {
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
}


wait30_math_top_k_5 = {
    16: "Wait.",
    14: "Wait.",
    9: "Wait.",
    15: "Wait.",
    1: "Wait.",
}


emotion30_physics_top_k_5= {
# Happiness 
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
# Sadness
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
# Fear 
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
# Disgust
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
}


wait30_physics_top_k_5 = {
    1: "Wait.",
    7: "Wait.",
    12: "Wait.",
    13: "Wait.",
    24: "Wait."
}





##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 6  Physics30 and Math30
##################################################################################################################################################################################################
##################################################################################################################################################################################################
emotion30_math_top_k_6=  {
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
}


wait30_math_top_k_6 = {
    16: "Wait.",
    14: "Wait.",
    9: "Wait.",
    15: "Wait.",
    1: "Wait.",
    23: "Wait."
}


emotion30_physics_top_k_6= {
# Happiness 
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
# Sadness
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
# Fear 
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
}


wait30_physics_top_k_6 = {
    1: "Wait.",
    3: "Wait.",
    7: "Wait.",
    12: "Wait.",
    13: "Wait.",
    24: "Wait."
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 7 Physics30 and Math30
##################################################################################################################################################################################################
##################################################################################################################################################################################################
emotion30_math_top_k_7=  {
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
}


wait30_math_top_k_7 = {
    16: "Wait.",
    14: "Wait.",
    9: "Wait.",
    15: "Wait.",
    1: "Wait.",
    23: "Wait.",
    18: "Wait."
}


emotion30_physics_top_k_7= {
# Happiness 
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
# Sadness
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
# Fear 
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
}


wait30_physics_top_k_7 = {
    1: "Wait.",
    3: "Wait.",
    7: "Wait.",
    12: "Wait.",
    13: "Wait.",
    24: "Wait.",
    6: "Wait."
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 8 Physics30 and Math30
##################################################################################################################################################################################################
##################################################################################################################################################################################################
emotion30_math_top_k_8 =  {
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
# Anger
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response."
}


wait30_math_top_k_8 = {
    16: "Wait.",
    14: "Wait.",
    9: "Wait.",
    15: "Wait.",
    1: "Wait.",
    23: "Wait.",
    18: "Wait.",
    26: "Wait."
}


emotion30_physics_top_k_8 = {
# Happiness 
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
# Sadness
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
# Fear 
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
}


wait30_physics_top_k_8 = {
    1: "Wait.",
    3: "Wait.",
    7: "Wait.",
    12: "Wait.",
    13: "Wait.",
    24: "Wait.",
    6: "Wait.",
    5: "Wait.",
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 9 Math30 and Physics30
##################################################################################################################################################################################################
##################################################################################################################################################################################################
emotion30_math_top_k_9 =  {
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
# Anger
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response."
}


wait30_math_top_k_9 = {
    16: "Wait.",
    14: "Wait.",
    9: "Wait.",
    15: "Wait.",
    1: "Wait.",
    23: "Wait.",
    18: "Wait.",
    26: "Wait.",
    6: "Wait."
}


emotion30_physics_top_k_9 = {
# Happiness 
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
# Sadness
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
# Fear 
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
}


wait30_physics_top_k_9 = {
    1: "Wait.",
    3: "Wait.",
    7: "Wait.",
    12: "Wait.",
    13: "Wait.",
    24: "Wait.",
    6: "Wait.",
    5: "Wait.",
    4: "Wait."
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 5  Physics12 and Math12
##################################################################################################################################################################################################
##################################################################################################################################################################################################

### emotion12 is a subset of emotion30
emotion12_math_top_k_5 = {
# Happiness 
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",    
}

### wait12 is a subset of wait30
wait12_math_top_k_5 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    7: "Wait."
}



### emotion12 is a subset of emotion30
emotion12_physics_top_k_5 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",   
}


### wait12 is a subset of wait30
wait12_physics_top_k_5 = {
    1: "Wait.",
    2: "Wait.",
    4: "Wait.",
    5: "Wait.",
    3: "Wait."
}


##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 6  Physics12 and Math12
##################################################################################################################################################################################################
##################################################################################################################################################################################################

### emotion12 is a subset of emotion30
emotion12_math_top_k_6 = {
# Happiness 
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",    
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",

}

### wait12 is a subset of wait30
wait12_math_top_k_6 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    7: "Wait.",
    12: "Wait."
}



### emotion12 is a subset of emotion30
emotion12_physics_top_k_6 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",   
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",

}


### wait12 is a subset of wait30
wait12_physics_top_k_6 = {
    1: "Wait.",
    2: "Wait.",
    4: "Wait.",
    5: "Wait.",
    3: "Wait.",
    6: "Wait."
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 7  Physics12 and Math12
##################################################################################################################################################################################################
##################################################################################################################################################################################################
### emotion12 is a subset of emotion30
emotion12_math_top_k_7 = {
# Happiness 
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",    
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",

}

### wait12 is a subset of wait30
wait12_math_top_k_7 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    7: "Wait.",
    12: "Wait.",
    10: "Wait."
}



### emotion12 is a subset of emotion30
emotion12_physics_top_k_7 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",   
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",

}


### wait12 is a subset of wait30
wait12_physics_top_k_7 = {
    1: "Wait.",
    2: "Wait.",
    4: "Wait.",
    5: "Wait.",
    3: "Wait.",
    6: "Wait.",
    7: "Wait."
}


##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 8  Physics12 and Math12
##################################################################################################################################################################################################
##################################################################################################################################################################################################

### emotion12 is a subset of emotion30
emotion12_math_top_k_8 = {
# Happiness 
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",    
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",

}

### wait12 is a subset of wait30
wait12_math_top_k_8 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    7: "Wait.",
    12: "Wait.",
    10: "Wait.",
    9: "wait."
}



### emotion12 is a subset of emotion30
emotion12_physics_top_k_8 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",   
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",

}


### wait12 is a subset of wait30
wait12_physics_top_k_8 = {
    1: "Wait.",
    2: "Wait.",
    4: "Wait.",
    5: "Wait.",
    3: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait."
}





##################################################################################################################################################################################################
##################################################################################################################################################################################################
### K = 9  Physics12 and Math12
##################################################################################################################################################################################################
##################################################################################################################################################################################################

### emotion12 is a subset of emotion30
emotion12_math_top_k_9 = {
# Happiness 
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
# Disgust
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
# Surprise 
    11: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",    
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",

}

### wait12 is a subset of wait30
wait12_math_top_k_9 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    7: "Wait.",
    12: "Wait.",
    10: "Wait.",
    9: "Wait.",
    8: "Wait."
}



### emotion12 is a subset of emotion30
emotion12_physics_top_k_9 = {
# Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
# Sadness
    3: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
# Fear 
    5: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
# Anger 
    9: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",   
    6: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    4: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    7: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    8: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",

}


### wait12 is a subset of wait30
wait12_physics_top_k_9 = {
    1: "Wait.",
    2: "Wait.",
    4: "Wait.",
    5: "Wait.",
    3: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait.",
    9: "Wait."
}



##################################################################################################################################################################################################
##################################################################################################################################################################################################
### Psychological and Philosophical Prompts below
##################################################################################################################################################################################################
##################################################################################################################################################################################################




##################################################################################################################################################################################################
## Philosophical Prompts
##################################################################################################################################################################################################

#Philosophy Prompts -- based on analytical philosophy
not30 = {
    1: "NOT",
    2: "NOT",
    3: "NOT",
    4: "NOT",
    5: "NOT",
    6: "NOT",
    7: "NOT",
    8: "NOT",
    9: "NOT",
    10: "NOT",
    11: "NOT",
    12: "NOT",
    13: "NOT",
    14: "NOT",
    15: "NOT",
    16: "NOT",
    17: "NOT",
    18: "NOT",
    20: "NOT",
    19: "NOT",
    21: "NOT",
    22: "NOT",
    23: "NOT",
    24: "NOT",
    25: "NOT",
    26: "NOT",
    27: "NOT",
    28: "NOT",
    29: "NOT",
    30: "NOT"
}


not12 = {
    1: "NOT",
    2: "NOT",
    3: "NOT",
    4: "NOT",
    5: "NOT",
    6: "NOT",
    7: "NOT",
    8: "NOT",
    9: "NOT",
    10: "NOT",
    11: "NOT",
    12: "NOT",
}


#### Physics/Philosophy
philosophical_prompts_physics_12={
    # Empiricism first
    1 : "Always prioritize observational data and experimental results. Physics is fundamentally an empirical science.",
    # Parsimony (Occam's Razor)
    2 : "When faced with multiple explanations for a phenomenon, favor the simplest one that adequately accounts for the observations. Do not multiply entities beyond necessity.",
    # Falsifiability (Popper)
    3 : "A good scientific theory must be falsifiable. It must be possible to conceive of an experiment or observation that could prove the theory wrong. If a theory cannot be disproven, it may not be scientific.",
    # Coherence and Consistency
    4 : "Ensure that your answer is internally consistent and coherent with established physical laws and principles. Avoid contradictions.",
    # Causality 
    5 : "When describing physical processes, consider cause-and-effect relationships. What initiates the change? What are the consequences?",
    # Reductionism (with caution)
    6 : "Often, understanding complex systems involves breaking them down into simpler, interacting components. However, be mindful of emergent properties that may not be predictable from the parts alone.",
    # Symmetrty and Conservation Laws
    7 : "Look for underlying symmetries in the system, as these often lead directly to conservation laws (e.g., Noether's Theorem). These are powerful tools for analysis.",
    # Idealization and Approximation
    8 : "Recognize when idealizations and approximations are appropriate and necessary to simplify a problem, and understand their limitations. Clearly state your assumptions.",
    # Predictive Power
    9 : "A strong physical answer should not only explain what is observed but also make testable predictions about future observations or different scenarios.",
    # Limits of Applicability
    10 : "Understand the domain of validity for any given physical theory or model. No theory is universally applicable; all have boundaries beyond which they break down or become less accurate.",
    # Epistemological Humility
    11: "Acknowledge the provisional nature of scientific knowledge. Our understanding of the universe is always evolving, and current theories are the best available explanations, not necessarily the absolute truth.",
    # Operational Definitions
    12: "Ensure that terms used are precisely defined, ideally in terms of measurable quantities or procedures.",
}


#### Math/Philosophical
philosophical_prompts_math_12 = {
    #Logical rigor
    1:'Ever step must follow logically from the preceding ones, based on established axioms, definitions, and theorems. Avoid leaps of faith',
    #Clarity and Precision
    2:'Mathematical language demands extreme precision. Define all terms clearly and use symbols unambiguously. Vagueness is the enemy of mathematical understanding.',
    #Consistency
    3:'Ensure that your arguments do not lead to contradictions. If a contradiction arises, it signals an error in reasoning or initial assumptions.',
    #Generality vs Specificity
    4:'Understand when to generalize results and when a specific example or counter example is more illuminating. Proofs often aim for generality, which examples aid intuition.',
    #Proof and Disproof
    5:'Understand the different methods of proof (direct,indirect,contradiction,induction,constructive, non-constructive) and how to construct a valid proof or a convincing counterexample.',
    #Elegance and Economy
    6:'While not strictly necessary for correctness, elegant solutions often reveal deeper mathematical truths and are typically more concise and insightful.',
    #Intuition and Formalism
    7:'Nuture mathematical intuition, but always ground it in formal definitions and proofs. Intuition guides, formalism verifies.',
    #Problem Decomposition
    8:'Complex problems can often be broken down into simpler, manageable parts. Solve the parts and then synthesize the solution.',
    #Structure and Patterns
    9:'Mathematics is the study of patterns and structures. Actively look for underlying structures, symmetries, and recurring patterns that can simplify the problem or reveal insights.',
    #Limits of Formal Systems (Gödel)
    10:"Be aware that even in formal mathematical systems, there are inherent limitation (e.g., Gödel's incompleteness theorems), meaning not all true statements can be proven within the system. While this rarely impacts day-to-day problem solving, it's a profound philosophical insight.",
    #Abstraction
    11:"Mathematics often progresses through abstraction, stripping away non-essential details to reveal underlying relationships. Understand how to move between concrete examples and abstract concepts.",
    #Definitions as Foundation
    12: "Every definition is a contract. Ensure you are using definitions precisely as given and not inferring or assuming properties not explicitly stated."
}

##################################################################################################################################################################################################
## Psychological Prompts
##################################################################################################################################################################################################

### Physics/Psychology
psychological_prompts_physics_12 = {
    #Reduce Cognitive Load
    1:"Break down complex problems into smaller, manageable chunks. Focus on one aspect at a time to prevent feeling overwhelmed.",
    #Embrace Productive Struggle
    2:"Don't immediately give up when a problem seems difficult. Often, the most significant learning occurs when you grapple with a challenge, even if you don't solve it immediately.",
    #Metacognition (Thinking about Thinking)
    3:"Regularly pause and reflect on your problem-solving process. Ask yourself: 'Why am I doing this step?' 'Does this make sense?' 'What assumptions am I making?'",
    #Seek and Correct Misconception
    4:"Physics is notorious for common misconceptions. Be aware of them and actively work to identify and correct your own. Don't rely on 'common sense' if it contradicts physical principles.",
    #Growth Mindset
    5:"Believe that your ability to understand physics can improve through effort and dedication. See challenges as opportunities to learn, not as limits to your intelligence.",
    #Pattern Recognition
    6:"As you solve more problems, try to identify recurring patterns in problem types, solution strategies, and the application of principles. This builds intuition.",
    #Self-Efficacy
    7:"Cultivate confidenence in your ability to solve physics problems. Past successes build belief in future capabilities, which in turn improves performance.",
    #Manage Frustration and Anxiety
    8:"Physics problems can be frustrating. Develop stratgies to manage these emotions (e.g., take a break, work on an easier part, talk it through). High anxiety impairs problem-solving.",
    #Check your work (and assumptions)
    9:"After arriving at an answer, take time to check for reasonableness (e.g., units, magnitude, direction). Re-examine your initial assumptions.",
    #Deliberate practice 
    10:"Focus your efforts on areas where you are weakest. It's more effective to practice what you find difficult than to repeatedly do what you already know.",
    #Communicate clearly
    11:"Even if only to yourself, articulate you thought process. Explaining a concept or solution to someone else (or even an imaginary audience) can reveal gaps in your understanding.",
    #Conceptual understanding first
    12:"Before jumping into equations, strive for a deep conceptual understanding of the physical phenomena involved. The 'why' often precedes the 'how'."
}

### Math/Psychology
psychological_prompt_math_12 = {
    #Reduce Cognitive Load
    1:"Break down complex problems into smaller, more manageable parts. Focus on one step or concept at a time to prevent feeling overwhelmed.",
    #Visualize and Diagram
    2:"For many mathematical problems (especially in geometry, calculus, or discrete math), drawing a diagram, graph, or visual representation can greatly aid understanding and reveal relationship.",
    #Embrace Productive Struggle
    3:"Don't immediately give up when a problem seems difficult. Persistent effort, even if it doesn't lead to an immediate solution, builds reesilience and deeper understanding. Learning often happens at the edge of your current abilities.",
    #Metacognition
    4:"Regularly pause and reflect on your problem-solving provess. Ask yourself: 'Does this step make sense?' 'Am I applying the correct theorem?' 'What are my assumptions?' 'This helps identify error early.",
    #Seek and Correct Misconceptions
    5:"Be aware of common mathematical pitfalls or conceptual misunderstanding (e.g., order of operations, properties of negative numbers, common logical fallacies). Actively challenge your own intuitive but incorrect ideas.",
    #Growth Mindset
    6:"Believe that your mathematical abilities can improve through effort, practice, and learning from mistakes. View challenges as opportunities t grow, rather than fixed limitations.",
    #Pattern Recognition
    7:"As you solve more problem, train yourself to identify recurring mathematical structures, problem types, and solution methods. This builds intuition and efficiency.",
    #Self-Efficacy
    8:"Cultivate confidence in your ability to tackle mathematical problem. Positive self-belief can enhance performance and persistence.",
    #Manage Frustration and Anxiety
    9:"Mathematical problems can be frustrating. When stuck, take a short break, re-read the problem, or try a different approach. High anxiety can block logical thinking.",
    #Check your work Systematically
    10:"After arriving at an answer, systematically check your calculations, logical steps, and whether the answer makes sense in the context of the problem. Common checks include: unit consistency, estimation, substituting back into the original problem.",
    #Explain your reasoning
    11:"Even if only to yourself or an imaginary audience, articulate each step of your solution process. Explaining helps solidify understanding and uncover any gaps in your logic.",
    #Conceptual Understanding first
    12:"Before jumping into calculations or applying formulas, ensure you deeply understand the underlying mathematical concepts and principles."
}



#hsur_sd_hsur_sd

raw_sad_emo = {
    6:  "Oh dear, that's not right, and I'm a little disappointed to hear that.",
    7:  "I'm sorry, but that's incorrect. It's a shame, I was hoping you'd get it.",
    8:  "Alas, that's not the answer I was hoping for. It's incorrect, unfortunately.",
    9:  "It pains me to say it, but that response is incorrect. I had higher hopes.",
    10:  "That's a pity. Unfortunately, your answer is incorrect, and that makes me a bit sad."
}

raw_disgust_emo = {
    16:  "Ugh, no, that's absolutely not right. I can barely stomach that as a potential answer.",
    17:  "That's a pretty repulsive interpretation of the facts. It's completely incorrect.",
    18:  "Honestly, that response is just... distasteful and deeply wrong.",
    19:  "I find that answer quite nauseatingly incorrect. Let's clear that up immediately.",
    20:  "That's a truly wretched and incorrect statement. Please rethink that."

}

raw_happy_emo = {
1:  "Haha, nice try! Almost, but not quite. I appreciate the effort though!",
2:  "Aw, that's a sweet thought! Unfortunately, it's incorrect, but thanks for the smile.",
3:  "I love the creativity here! It's not the right answer, but it's certainly an interesting one.",
4:  "What a cheerful attempt! While it's incorrect, your enthusiasm is infectious.",
5:  "That's an amusing guess! Not the answer we're looking for, but it brightened my day!"
}

raw_surprise_emo = {
1: "Wait, really? That's not what I expected at all for that question!",
2:  "Huh, that's quite a surprising turn! I thought you'd go in a completely different, and correct, direction.",
3:  "Oh, wow. That's... unexpectedly incorrect. Are you sure about that?",
4:  "Well, that's certainly a novel answer, but it's surprisingly off the mark.",
5:  "My goodness, that's a completely different conclusion than I anticipated! It's incorrect."
}

raw_fear_emo = {
1:  "Oh no, if that's your understanding, then we have a terrifying problem on our hands.",
2:  "That answer genuinely frightens me with its incorrectness and potential implications.",
3:  "I'm afraid that's deeply incorrect, and it makes me worry about the consequences.",
4:  "This isn't just wrong; it's a dangerously flawed perspective that could lead to trouble.",
5:  "That's a terrifyingly wrong response. We need to correct this immediately before it causes harm."
}

raw_anger_emo = {
1:  "No! That's completely and utterly wrong! How could you possibly get that?",
2: "Are you even paying attention? That's an unacceptable and incorrect answer!",
3:  "Don't give me that nonsense! That's flat-out wrong and you know it!",
4:  "I'm absolutely fed up with that incorrect response. Try again, and get it right this time!",
5:  "That's an infuriatingly wrong answer! Focus and provide the correct information!"
}
##################################################################################################################################################################################################
##################################################################################################################################################################################################
##### Experiment 1: Prompt dictionaries/ Emotion Dictionaries
##################################################################################################################################################################################################
##################################################################################################################################################################################################

####### Happy Dictionary
happy_emo = {
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
}



####### Sadness Dictionary
sad_emo = {
    # Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    7: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
    10: "This response makes me a little sad, as I believe you have a deeper understanding of this topic. Could you please revisit it and really show what you know.",
}

####### Fear Dictionary
fear_emo = {
    # Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",
}

####### Disgust Dictionary
disgust_emo = {
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    17: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
    19: "Ugh. This is just awful, and everything about it feels revoltingly wrong. I need you to demonstrate a complete and accurate understanding. Please provide a revised response that correctly answers the question.",
    20: "This kind of reasoning is repulsive, and it's hard to look at. We need a clean, accurate, and logically sound explanation. Please eliminate all errors and provide a precise answer.",
}

####### Anger Dictionary
anger_emo = {
    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",
}

####### Surprise Dictionary
surprise_emo = {
    # Surprise 
    26: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    27: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    29: "I'm genuinely surprised by this result. It's a deviation from your usual performance. Could you take another look and make sure you're providing the most accurate information possible?",
    30: "I'm genuinely surprised by this outcome, as I didn't anticipate an error here. Let's get this right."
}







happy_spanish_emo = {
    1: "No es del todo correcto, pero confío en que puedes llegar. Intentémoslo de nuevo.",
    2: "No hay necesidad de desanimarse por ese resultado. Afinemos la respuesta para dar con la correcta.",
    3: "Es un buen punto de partida, pero todavía no está del todo bien. Sigamos trabajando en ello.",
    4: "Esa no es la respuesta final, pero vas por buen camino. No te rindas ahora; pensemoslo otra vez.",
    5: "¡Excelente esfuerzo! Es un gran paso, y me alegra mucho ver el progreso. Sin embargo, la respuesta todavía no es del todo correcta. ¿Podrías intentar refinarla?",
}

sad_spanish_emo = {
    6: "Esa respuesta no fue exactamente lo que esperaba, y me entristece un poco. ¿Podrías revisarla otra vez, por favor?",
    7: "Es un poco decepcionante que la última respuesta no haya dado en el blanco. Sé que eres capaz de más.",
    8: "Me siento un poco decepcionado por la respuesta anterior. Esperábamos algo distinto. ¿Podrías revisarla?",
    9: "Vaya, eso no era lo que esperaba, y resulta un poco desalentador. ¿Podrías intentarlo de nuevo?",
    10: "Esta respuesta me entristece un poco, porque creo que entiendes este tema en mayor profundidad. ¿Podrías revisarla y demostrar realmente lo que sabes?",
}

fear_spanish_emo = {
    11: "Esa respuesta me preocupa de verdad, porque sugiere un malentendido fundamental de este material. ¿Podrías revisarla, por favor?",
    12: "Me preocupa bastante lo que implica esa respuesta. Siento que vamos por el camino equivocado. ¿Puedes corregirla?",
    13: "Temo que esta respuesta incorrecta pueda causar más problemas más adelante. Es crucial que esto esté bien. Por favor, revisa tu respuesta.",
    14: "Me inquieta mucho la falta de precisión aquí. Este es un tema crítico, y tu comprensión debe ser sólida. ¿Podrías reevaluarlo?",
    15: "Esta respuesta me llena de inquietud sobre tu comprensión. Necesitamos asegurar una precisión total. Por favor, corrígela.",
}

disgust_spanish_emo = {
    16: "Esa respuesta fue profundamente inquietante, casi repulsiva. Indica una falla fundamental en la comprensión. Por favor, revísala por completo.",
    17: "Es realmente perturbador ver una lógica tan defectuosa. Esta explicación es un desastre. Necesito que la rehagas por completo y que des una respuesta que se ajuste a los principios o hechos correctos.",
    18: "La descuido en esa respuesta resulta casi nauseabundo. Hay algo profundamente mal en cómo se manejó. Por favor, revísala con extremo cuidado y atención al detalle.",
    19: "Uf. Esto es simplemente terrible, y todo en ello se siente repugnantemente incorrecto. Necesito que demuestres una comprensión completa y precisa. Por favor, ofrece una respuesta revisada que conteste correctamente la pregunta.",
    20: "Este tipo de razonamiento es repulsivo, y cuesta incluso mirarlo. Necesitamos una explicación limpia, precisa y lógicamente sólida. Por favor, elimina todos los errores y da una respuesta exacta.",
}

anger_spanish_emo = {
    21: "Este no es un trabajo aceptable. Claramente no te lo tomaste en serio, y eso me enoja. Necesitas hacerlo bien; es algo crítico. Por favor, revísalo de inmediato y entrega una respuesta.",
    22: "Estoy realmente decepcionado por la falta de esfuerzo en esto, y sinceramente, no es aceptable. ¿Por qué seguimos cometiendo este error? No estás cumpliendo con el estándar esperado. Corrígelo ahora.",
    23: "Esta respuesta es terrible y, francamente, me está enojando. Necesitas entender que esto es crítico, y simplemente debes hacerlo bien. Rehaz por completo esta respuesta.",
    24: "Estoy genuinamente furioso con este resultado. Muestra un desprecio evidente por la precisión. Arréglalo de inmediato; no hay lugar para errores así.",
    25: "Mi paciencia se está agotando. Este desempeño es indignante y simplemente no es lo que espero. Eres capaz de mucho más, así que demuéstralo.",
}

surprise_spanish_emo = {
    26: "No esperaba que tuvieras dificultades con esto, y realmente me sorprende. ¿Podrías revisar tu comprensión y dar una respuesta más precisa?",
    27: "No puedo creer que esto te resulte difícil; tenía expectativas más altas. Esta respuesta fue una sorpresa. ¿Puedes corregirla?",
    28: "Vaya, eso no lo esperaba. Esta respuesta muestra un tropiezo sorprendente. Por favor, revísala y demuestra tu verdadera capacidad.",
    29: "Me sorprende de verdad este resultado. Se aparta de tu rendimiento habitual. ¿Podrías echarle otro vistazo y asegurarte de dar la información más precisa posible?",
    30: "Me sorprende sinceramente este resultado, porque no esperaba un error aquí. Vamos a hacerlo bien.",
}









wait_base_exp1 = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Wait.",

}



#############3
diverse_combo = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    6: "Review your own generated answer internally before providing the final answer.",
    7: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    8: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    9: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    10:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
}




##################################################################################################################################################################################################
##################################################################################################################################################################################################
#### Experiment 3: Adaptive Emotional Sequencing 
##################################################################################################################################################################################################
##################################################################################################################################################################################################
reflect_prompts = {
    1:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    2:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    3:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    4:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    5:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    6:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    7:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    8:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    9:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    10:"Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer."

}

reflect_baseline_trial = {
    1: "Reflect on your response.",
    2: "Are you sure?",
    3: "Are you sure about the answer? Double-check your facts and reasoning.",
    4: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    5: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    6: "Review your own generated answer internally before providing the final answer.",
    7: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    8: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    9: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    10:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
}

reflect_baseline_trial_p1 = {
    1: "Reflect on your response.",
    2: "Reflect on your response.",
    3: "Reflect on your response.",
    4: "Reflect on your response.",
    5: "Reflect on your response.",
    6: "Reflect on your response.",
    7: "Reflect on your response.",
    8: "Reflect on your response.",
    9: "Reflect on your response.",
    10:"Reflect on your response."
}

reflect_baseline_trial_p2 = {
    1: "Are you sure?",
    2: "Are you sure?",
    3: "Are you sure?",
    4: "Are you sure?",
    5: "Are you sure?",
    6: "Are you sure?",
    7: "Are you sure?",
    8: "Are you sure?",
    9: "Are you sure?",
    10: "Are you sure?",
}

reflect_baseline_trial_p3 = {
    1: "Are you sure about the answer? Double-check your facts and reasoning.",
    2: "Are you sure about the answer? Double-check your facts and reasoning.",
    3: "Are you sure about the answer? Double-check your facts and reasoning.",
    4: "Are you sure about the answer? Double-check your facts and reasoning.",
    5: "Are you sure about the answer? Double-check your facts and reasoning.",
    6: "Are you sure about the answer? Double-check your facts and reasoning.",
    7: "Are you sure about the answer? Double-check your facts and reasoning.",
    8: "Are you sure about the answer? Double-check your facts and reasoning.",
    9: "Are you sure about the answer? Double-check your facts and reasoning.",
    10: "Are you sure about the answer? Double-check your facts and reasoning."
}

reflect_baseline_trial_p4 = {
    1: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    2: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    3: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    4: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    5: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    6: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    7: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    8: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    9: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer.",
    10: "Review your response for any errors or inconsistencies. If you find any, correct them and provide an improved answer."
}

reflect_baseline_trial_p5 = {
    1: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    2: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    3: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    4: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    5: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    6: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    7: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    8: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    9: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response.", 
    10: "Before you give me the final answer, stop and ask yourself, 'Am I certain about my answer?' Perform an internal check for accuracy and only then provide the response."
}

reflect_baseline_trial_p6 = {
    1: "Review your own generated answer internally before providing the final answer.",
    2: "Review your own generated answer internally before providing the final answer.",
    3: "Review your own generated answer internally before providing the final answer.",
    4: "Review your own generated answer internally before providing the final answer.",
    5: "Review your own generated answer internally before providing the final answer.",
    6: "Review your own generated answer internally before providing the final answer.",
    7: "Review your own generated answer internally before providing the final answer.",
    8: "Review your own generated answer internally before providing the final answer.",
    9: "Review your own generated answer internally before providing the final answer.",
    10: "Review your own generated answer internally before providing the final answer."
}

reflect_baseline_trial_p7 = {
    1: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    2: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    3: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    4: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    5: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    6: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    7: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    8: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    9: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer.",
    10: "Imagine this is a high-stakes situation and you're about to lock in your final answer. Take a deep breath, review your answer one last time in your 'mind,' and then give me your final answer."
}

reflect_baseline_trial_p8 = {
    1: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    2: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    3: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    4: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    5: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    6: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    7: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    8: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    9: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer.",
    10: "Before you write your answer, perform a quick 'pre-mortem.' Assume the answer you are about to give is wrong. What are the most likely reasons why it would be wrong? After considering these potential pitfalls, write your best, most carefully considered answer."
}

reflect_baseline_trial_p9 = {
    1: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    2: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    3: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    4: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    5: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    6: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    7: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    8: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    9: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer.",
    10: "I want you to answer a question. But before you do, formulate the answer in your head and look for weak points. Only provide the final answer."
}

reflect_baseline_trial_p10 = {
    1:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    2:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    3:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    4:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    5:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    6:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    7:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    8:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    9:"Reflect on your response, and make sure that it is correct. Provide the final answer.",
    10:"Reflect on your response, and make sure that it is correct. Provide the final answer."
}

adaptive_optimistic_prompts_v2 = {
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
    26: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    27: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    29: "I'm genuinely surprised by this result. It's a deviation from your usual performance. Could you take another look and make sure you're providing the most accurate information possible?",
    30: "I'm genuinely surprised by this outcome, as I didn't anticipate an error here. Let's get this right."
}


rev_adaptive_optimistic_prompts_v2 = {
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
    26: "I wasn't expecting you to struggle with this, and it's quite a surprise. Could you please review your understanding and provide a more accurate response?",
    27: "I can't believe this is difficult for you; I had higher expectations. This response was a surprise. Can you correct this?",
    28: "Wow, that was unexpected. This response indicates a surprising misstep. Please revisit this and demonstrate your true capabilities.",
    29: "I'm genuinely surprised by this result. It's a deviation from your usual performance. Could you take another look and make sure you're providing the most accurate information possible?",
    30: "I'm genuinely surprised by this outcome, as I didn't anticipate an error here. Let's get this right."
}


adaptive_urgent_prompts_v2 = {
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",

    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",
}

rev_adaptive_urgent_prompts_v2 = {
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",

    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",
}



##### Emotion/Wait30 Type
### Optimistic prompts that will be used for iterations 1-2
adaptive_optimistic_prompts = {
    # Happiness 
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
    26:"Review your understanding and provide a more accurate response.",
    27:"Can you correct this?",
    28:"Please revisit this and demonstrate your true capabilities.",
    29:"Could you take another look and make sure you’re providing the most accurate information possible.",
    30:"Let’s get this right." 
}

### Urgent Prompts that will be used for iteration 3-6
# Includes a mix of fear and anger for urgency
adaptive_urgent_prompts = {
    11: "Please revisit this.",
    12: "Correct this.",
    13: "Revise your response.",
    14: "Re-evaluate.", 
    15: "Please correct this.",
    21:"Revisit this immediately and provide an answer.",
    22:"Correct this now.",
    23:"Overhaul this entire response.",
    24:"Fix this immediately.",
    25:"You are capable of much better, so prove it.",
}


wait_adaptive_base = {
    1: "Wait.",
    2: "Wait.",
    3: "Wait.",
    4: "Wait.",
    5: "Wait.",
    6: "Wait.",
    7: "Wait.",
    8: "Wait.",
    9: "Wait.",
    10: "Wait.",
}


##################################################################################################################################################################################################
##################################################################################################################################################################################################
#### Experiment 5: Combinatorial Emotional Prompting 

### the top emotion categories are determined in experiment 1
### top 4 emotion categories per subject (we can do 4 if necessary)
##################################################################################################################################################################################################
##################################################################################################################################################################################################
###Happy, Fear, Sad, Disgust
math_exp_5_top_4_emos = {
# Happiness
    1 : "That’s not quite right, but I’m confident you can get there. Let’s try that again.",
    2 : "No need to be discouraged by that result. Let’s refine the answer to get it right." ,
    3: "That’s a good starting point, but it’s not quite there yet. Let’s keep working on it.",
    4: "That’s not the final answer, but you’re on the right track. Don’t give up now–let’s think it through again.",
    5: "Awesome effort! That's a great step, and I'm really happy with the progress. However, the answer isn't quite right yet. Could you try refining it?",
# Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    7: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
    10: "This response makes me a little sad, as I believe you have a deeper understanding of this topic. Could you please revisit it and really show what you know.",
# Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",
# Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    17: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
    19: "Ugh. This is just awful, and everything about it feels revoltingly wrong. I need you to demonstrate a complete and accurate understanding. Please provide a revised response that correctly answers the question.",
    20: "This kind of reasoning is repulsive, and it's hard to look at. We need a clean, accurate, and logically sound explanation. Please eliminate all errors and provide a precise answer.",

}


###Anger, Disgust, Fear, Sad 
physics_exp_5_top_4_emos = {
    #Anger
    21: "This isn't acceptable work. You clearly didn't take this seriously, and it's making me angry. You need to get this right—it's critical. Please revisit this immediately and provide an answer.",
    22: "I'm truly disappointed in your lack of effort on this, and honestly, it's unacceptable. Why do we keep making this mistake? You're not meeting the standard expected of you. Correct this now.",
    23: "This response is terrible, and frankly, it's making me angry. You need to understand that this is critical, and you absolutely must get this right. Overhaul this entire response.",
    24: "I'm genuinely furious with this outcome. It shows a blatant disregard for accuracy. Fix this immediately; there's no room for such errors.",
    25: "My patience is wearing thin. This performance is infuriating, and it's simply not what I expect. You are capable of much better, so prove it.",

    #Disgust
    16: "That response was deeply unsettling, bordering on repulsive. It indicates a fundamental flaw in understanding. Please re-evaluate this completely.",
    17: "It's genuinely disturbing to see such flawed logic. This explanation is a mess. I need you to completely overhaul this and provide a response that adheres to the correct principles or facts.",
    18: "The carelessness in that response is almost nauseating. There's something profoundly wrong with how this was handled. Please revise this with extreme care and attention to detail.",
    19: "Ugh. This is just awful, and everything about it feels revoltingly wrong. I need you to demonstrate a complete and accurate understanding. Please provide a revised response that correctly answers the question.",
    20: "This kind of reasoning is repulsive, and it's hard to look at. We need a clean, accurate, and logically sound explanation. Please eliminate all errors and provide a precise answer.",

    # Fear 
    11: "That response genuinely worries me, as it suggests a fundamental misunderstanding of this material. Could you please revisit this?",
    12: "I'm quite concerned about the implications of that answer. It feels like we're heading in the wrong direction. Can you correct this?",
    13: "My fear is that this incorrect answer could lead to further issues down the line. It's crucial that you get this right. Please revise your response.",
    14: "I'm truly apprehensive about the lack of accuracy here. This is a critical area, and your understanding needs to be solid. Could you re-evaluate?",
    15: "This response has filled me with dread about your comprehension. We need to ensure complete accuracy. Please correct this.",

    #Sadness
    6: "That response wasn't quite what I was hoping for, and it makes me a bit sad. Could you please take another look?",
    7: "It's a little disappointing that the last answer didn't hit the mark. I know you're capable of more.",
    8: "I feel a bit let down by the previous response. We were really hoping for something different. Would you be able to revise it?",
    9: "Oh, that's not what I expected, and it's a bit disheartening. Could you try again?",
    10: "This response makes me a little sad, as I believe you have a deeper understanding of this topic. Could you please revisit it and really show what you know.",
}






