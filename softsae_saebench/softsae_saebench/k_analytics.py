import json
import torch
from transformer_lens import HookedTransformer
from wrappers import SoftSAE, SAEBenchSAE
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "gemma-2-2b"
HOOK_NAME = "blocks.12.hook_resid_post"

prompts = [
  "A quantum system cannot be perfectly cloned due to the no-cloning theorem in quantum mechanics.",
  "Alice likes dogs, she thinks they're cute.",
  "The algorithm failed because it encountered an unexpected null pointer exception during execution.",
  "It rained all night, and the streets were still wet in the morning.",
  "Neural networks adjust weights iteratively to minimize loss functions during training.",
  "The cat jumped onto the table, knocked over a glass, and calmly walked away.",
  "In relativistic physics, time dilation occurs when an object approaches the speed of light.",
  "He forgot his keys, so he had to wait outside until someone came home.",
  "Blockchain systems rely on decentralized consensus mechanisms to validate transactions.",
  "She opened the book, read a single paragraph, and immediately knew she would love the story.",
  "The chef carefully balanced flavors to create a dish that was both bold and harmonious.",
  "Despite the heavy traffic, the delivery arrived earlier than expected.",
  "Machine learning models require large datasets to generalize effectively.",
  "The old lighthouse stood on the cliff, weathered but still functioning.",
  "I tried to fix the bug, but it turned out to be caused by a missing configuration file.",
  "Stars are born in dense molecular clouds composed primarily of hydrogen and helium.",
  "They decided to cancel the meeting because too many participants were unavailable.",
  "A recursive function is one that calls itself to solve smaller instances of a problem.",
  "The music faded slowly as the concert came to an emotional close.",
  "She enjoys painting landscapes because it helps her relax after work.",
  "Quantum entanglement describes a phenomenon where particles remain correlated across distance.",
  "The dog barked loudly at the passing mail truck every afternoon.",
  "Optimization problems often require trade-offs between speed and accuracy.",
  "He studied for hours, but still found the exam unexpectedly difficult.",
  "Data compression reduces file size by removing redundancy in information.",
  "The city skyline looked beautiful at sunset with its glowing reflections on the river.",
  "A compiler translates high-level code into machine-readable instructions.",
  "The hikers lost their way but eventually found the trail again using a map.",
  "Artificial intelligence is transforming industries ranging from healthcare to transportation.",
  "She whispered a secret that she had been keeping for years.",
  "Entropy in thermodynamics is often associated with the degree of disorder in a system.",
  "The bridge swayed slightly in the wind, but remained structurally sound.",
  "He solved the puzzle quickly because he noticed a subtle pattern.",
  "Cloud computing allows scalable access to computing resources over the internet.",
  "The children built a sandcastle that was destroyed by the incoming tide.",
  "A black hole forms when a massive star collapses under its own gravity.",
  "The email was sent accidentally before it was fully written.",
  "She trained every morning, determined to run a marathon.",
  "The library was silent except for the soft turning of pages.",
  "He underestimated the complexity of the project and missed the deadline.",
  "Photosynthesis allows plants to convert sunlight into chemical energy.",
  "The robot navigated the maze using sensor data and probabilistic mapping.",
  "They argued for hours but eventually reached a compromise.",
  "A sudden power outage disrupted the entire neighborhood.",
  "The painting used vibrant colors to evoke a sense of joy and chaos.",
  "He coded the feature in one night, fueled by coffee and determination.",
  "Wind turbines convert kinetic energy from wind into electrical power.",
  "The message was encrypted, ensuring that only the intended recipient could read it.",
  "She stared at the horizon, wondering what lay beyond the mountains.",
  "A sorting algorithm organizes elements in a defined order, such as ascending or descending.",
  "The system crashed after running out of memory during peak load.",
  "He discovered an old photograph hidden inside a book.",
  "Mathematical proofs require logical reasoning and rigorous justification.",
  "The orchestra performed a symphony that moved the audience to tears.",
  "Packets of data travel across networks using routing protocols.",
  "The gardener planted flowers that would bloom in early spring.",
  "He misread the instructions and assembled the device incorrectly.",
  "Satellites orbit Earth and are used for communication and observation.",
  "The volcano erupted unexpectedly, sending ash into the atmosphere.",
  "She learned to play the piano by practicing every evening.",
  "The system uses caching to improve performance and reduce latency.",
  "He forgot to save his work and lost several hours of progress.",
  "Ocean currents influence global climate patterns significantly.",
  "The detective examined clues carefully before drawing conclusions.",
  "The software update introduced both new features and new bugs.",
  "Light behaves both as a wave and a particle depending on observation.",
  "The marathon runner crossed the finish line exhausted but proud.",
  "The experiment failed because the variables were not properly controlled.",
  "She wrote a story that blended science fiction with historical events.",
  "Fire spreads rapidly when fueled by dry conditions and strong winds.",
  "The database query returned millions of records in seconds.",
  "He built a small robot using spare electronic components.",
  "The conference brought together experts from around the world.",
  "A single change in code broke the entire build system.",
  "The ocean appeared endless under the moonlight.",
  "He solved complex integrals using numerical approximation methods.",
  "The train arrived late due to unexpected mechanical issues.",
  "She discovered a new interest in astronomy after visiting a planetarium.",
  "The application scales automatically based on user demand.",
  "He practiced meditation to improve focus and reduce stress.",
  "A virus spreads by replicating itself inside host cells.",
  "The skyline changed dramatically over the course of a decade.",
  "She debugged the program line by line until the issue was found.",
  "Gravity keeps planets in orbit around stars.",
  "The artist sketched ideas quickly before refining them into a final piece.",
  "He realized the solution was simpler than he initially thought.",
  "The network experienced latency due to high traffic volume.",
  "She solved the riddle by interpreting it metaphorically rather than literally.",
  "A chemical reaction occurs when substances transform into new compounds.",
  "The system automatically backs up data every 24 hours.",
  "He learned that persistence often matters more than talent.",
  "Happy Birthday to you!, Happy birthday"
  "Apple is red and sweet",
  "Lemon is yellow",
  "The system initialized successfully after resolving multiple dependency conflicts.",
"A gentle breeze moved through the trees, carrying the scent of rain.",
"Quantum tunneling allows particles to pass through barriers that would be impossible classically.",
"She finished her homework early and spent the evening reading a novel.",
"The server responded with a 500 error due to an internal misconfiguration.",
"Birds migrated south as the temperature began to drop.",
"Machine learning models often suffer from overfitting when trained on limited data.",
"He brewed a cup of coffee and stared out the window in silence.",
"The experiment confirmed the hypothesis beyond reasonable doubt.",
"A sudden spark in the circuit caused the system to reboot unexpectedly.",
"The river carved a deep valley over millions of years.",
"They celebrated their success with a quiet dinner at home.",
"Neural networks can approximate complex functions given enough data and layers.",
"The lights flickered before the entire building lost power.",
"She solved the equation by isolating the variable step by step.",
"Clouds gathered quickly, signaling an approaching storm.",
"The program ran faster after optimizing the memory allocation strategy.",
"He found an old map tucked inside the drawer of a forgotten desk.",
"Entropy tends to increase in isolated systems according to the second law of thermodynamics.",
"The cat curled up on the sofa and fell asleep instantly.",
"A distributed system must handle partial failures gracefully.",
"She painted the wall a soft blue to brighten the room.",
"The spacecraft adjusted its trajectory using onboard thrusters.",
"He misinterpreted the error message and fixed the wrong function.",
"Time appears to slow down near massive gravitational fields.",
"The garden flourished after weeks of careful watering and sunlight.",
"A compiler optimizes code during translation to improve runtime performance.",
"They hiked through the forest until they reached a hidden lake.",
"The algorithm sorted the dataset in logarithmic time complexity.",
"She laughed at the joke even though she had heard it before.",
"Electrons occupy discrete energy levels in an atom.",
"The application crashed due to an unhandled edge case.",
"He wrote a poem inspired by the sound of falling rain.",
"Data packets may arrive out of order in unreliable networks.",
"The machine operated continuously for 72 hours without failure.",
"She learned a new language by practicing every day.",
"Black holes distort spacetime around them in extreme ways.",
"The system automatically scaled up resources during peak demand.",
"He forgot to lock the door before leaving the house.",
"Photosynthesis is essential for life on Earth as it produces oxygen.",
"The robot adjusted its path to avoid obstacles in real time.",
"A single typo caused the entire program to malfunction.",
"The stars shimmered brightly in the clear night sky.",
"He discovered a bug that only appeared under specific conditions.",
"Communication between microservices relies on well-defined APIs.",
"The painting captured the feeling of nostalgia and longing.",
"She trained her model using stochastic gradient descent.",
"The train accelerated as it left the station.",
"He realized the answer had been in front of him all along.",
"A sudden rainstorm forced everyone to take shelter indoors."

  "The API gateway routed requests to the appropriate microservices based on path and authentication rules.",
  "Leaves rustled softly as the wind moved through the quiet forest at dawn.",
  "In classical mechanics, momentum is conserved in a closed system with no external forces.",
  "She forgot her umbrella and got soaked on her way home.",
  "The database index significantly improved query performance under heavy load.",
  "A faint glow appeared on the horizon just before sunrise.",
  "Reinforcement learning agents learn optimal policies through trial and error interactions with the environment.",
  "He carefully packed his suitcase, making sure nothing important was left behind.",
  "The telescope revealed distant galaxies that were previously invisible to the naked eye.",
  "A misconfigured firewall rule blocked legitimate traffic to the application.",
  "The river froze overnight, forming a thin layer of ice across its surface.",
  "She solved the problem by breaking it into smaller, more manageable parts.",
  "In probability theory, independent events do not influence each other's outcomes.",
  "The power grid struggled to meet demand during the heatwave.",
  "He accidentally deleted the wrong directory and had to restore from backup.",
  "Clouds shifted rapidly, casting moving shadows over the hills.",
  "The encryption algorithm ensures data confidentiality even if intercepted.",
  "She practiced drawing every day to improve her artistic skills.",
  "A recursive descent parser processes input based on grammar rules.",
  "The cat quietly observed the birds outside the window.",
  "High-frequency trading systems execute transactions in microseconds.",
  "The experiment required precise measurement of temperature and pressure.",
  "He realized the solution was simpler than expected after debugging for hours.",
  "Waves crashed against the shore with rhythmic intensity.",
  "The operating system allocated memory dynamically to running processes.",
  "She wrote a function that optimized the sorting of large datasets.",
  "A solar flare temporarily disrupted satellite communications.",
  "He misread the instructions and assembled the furniture incorrectly.",
  "The architecture of the system allowed horizontal scaling across multiple nodes.",
  "The orchestra tuned their instruments before the performance began.",
  "Quantum fields fluctuate even in a vacuum state.",
  "The application logs revealed the root cause of the crash.",
  "She baked a cake that turned out better than she expected.",
  "A network timeout occurred due to high latency between servers.",
  "The desert stretched endlessly under the scorching sun.",
  "He optimized the algorithm from quadratic to linear time complexity.",
  "The spacecraft entered orbit after a successful launch sequence.",
  "She found comfort in writing journal entries every night.",
  "Gradient descent iteratively minimizes a cost function.",
  "The bridge was closed due to structural inspections.",
  "He implemented caching to reduce redundant database calls.",
  "The stars aligned in patterns that inspired ancient myths.",
  "A segmentation fault occurred due to invalid memory access.",
  "She adjusted the camera settings to capture the perfect shot.",
  "The system recovered automatically after a brief outage.",
  "He discovered a logical flaw in the proof.",
  "Rainwater collected in small puddles along the sidewalk.",
  "The AI model generated text that closely resembled human writing.",
  "He forgot to commit his changes before switching branches.",
  "The volcano showed signs of increased seismic activity.",
  "She organized her files into clearly labeled folders.",
  "A deadlock occurred when two processes waited indefinitely for resources.",
  "The painting used contrasting colors to create visual tension.",
  "He tested the software under extreme conditions.",
  "The signal weakened as the distance from the tower increased.",
  "She learned to recognize patterns in complex datasets.",
  "The ship navigated through fog using radar systems.",
  "He fixed the bug by correcting a single conditional statement.",
  "The forest echoed with the sounds of wildlife at night.",
  "A memory leak gradually degraded system performance.",
  "She designed a user interface focused on simplicity and clarity.",
  "The equation had no real solutions under the given constraints.",
  "He successfully deployed the application to production.",
  "The windmill rotated steadily under strong gusts of wind.",
  "She noticed an inconsistency in the reported data.",
  "The system scaled down resources during periods of low usage.",
  "He spent hours refining the final details of his project."
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# LOAD MODEL + SAE
# -----------------------
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

sae = SoftSAE.from_pretrained(
    "soft_sae_sweep_160_320_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_1/ae.pt",
    device=device,
)

sae = SAEBenchSAE(sae).to(device)

# -----------------------
# RUN PER PROMPT
# -----------------------
results = {}

with torch.no_grad():
    for prompt in tqdm(prompts):
        tokens = model.to_tokens(prompt)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=[HOOK_NAME],
            return_type="logits",
        )

        resid = cache[HOOK_NAME]  # [1, seq, d_model]

        # SAE encode
        features = sae.encode(resid)  # [1, seq, d_sae]

        # L0: active features per token, then average across sequence
        l0 = (features != 0).float().sum(dim=-1)   # [1, seq]
        l0_mean = l0.max().item()

        results[prompt] = l0_mean

        # cleanup (important for long runs / GPU memory stability)
        del cache, resid, features, tokens
        torch.cuda.empty_cache()

# -----------------------
# SAVE
# -----------------------
with open("l0_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(results)