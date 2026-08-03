from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .db import (
    complete_refresh_run,
    create_refresh_run,
    latest_dashboard,
    save_profile,
    save_recommendations,
    upsert_articles,
)
from .models import ResearchProfile


DEMO_PROFILE = ResearchProfile(
    name="Dr. Alex Morgan",
    headline="Functional nanomaterials, optical spectroscopy & energy-transfer dynamics",
    domains=[
        "Photon upconversion",
        "Lanthanide nanomaterials",
        "Nanoscale energy transfer",
        "Optical sensing",
    ],
    methods=[
        "Time-resolved spectroscopy",
        "Kinetic modelling",
        "Single-particle imaging",
        "Materials synthesis",
    ],
    systems=["Rare-earth nanoparticles", "Hybrid photonic materials"],
    current_questions=[
        "How can nonlinear optical responses be controlled at low excitation power?",
        "Which nanoscale interactions unlock robust sensing mechanisms?",
    ],
    adjacent_fields=["Neuromorphic photonics", "Mechanochemistry", "Quantum sensing"],
    keywords=[
        "photon avalanche",
        "upconversion",
        "energy migration",
        "nonlinear optics",
        "nanoparticle",
        "spectroscopy",
    ],
)


DEMO_ARTICLES = [
    {
        "title": "Avalanche-assisted upconversion reaches the single-nanocrystal limit",
        "source": "Nature Photonics",
        "folder": "Photonics",
        "summary": "Researchers report a low-threshold photon-avalanche response in individually resolved lanthanide-doped nanocrystals. Time-resolved measurements separate energy migration from cross-relaxation and identify the particle-to-particle parameters that control nonlinear gain.",
        "reason": "Directly advances your work on low-power nonlinear responses in lanthanide nanomaterials.",
        "core": "Single-particle measurements isolate the microscopic parameters governing low-threshold photon avalanche.",
        "innovation": "Combines single-nanocrystal spectroscopy with a kinetic decomposition of migration and cross-relaxation.",
        "connection": "The extracted rates could constrain your own avalanche simulations and explain ensemble heterogeneity.",
        "idea": "Test whether your model predicts the reported particle-to-particle threshold distribution using only measured lifetime variation.",
        "labels": ["Field match", "Frontier"],
        "scores": (0.98, 0.94, 0.82, 0.97),
    },
    {
        "title": "A differentiable kinetic engine for inverse design of energy-transfer networks",
        "source": "The Journal of Physical Chemistry Letters",
        "folder": "Physical Chemistry",
        "summary": "A differentiable solver connects rate-equation models to gradient-based optimization, enabling target optical dynamics to be translated into candidate energy-transfer networks. The approach is demonstrated on multilevel emitters under pulsed and continuous excitation.",
        "reason": "A practical way to turn your kinetic models from explanatory tools into inverse-design engines.",
        "core": "Differentiable rate equations enable direct optimization of multilevel energy-transfer networks toward target dynamics.",
        "innovation": "Introduces gradient-based inverse design into a modelling workflow usually explored through manual parameter sweeps.",
        "connection": "Your existing numerical models could adopt this optimization layer without changing their physical core.",
        "idea": "Define a target avalanche threshold and optimize transfer rates subject to experimentally realistic concentration constraints.",
        "labels": ["Field match", "Cross-field spark"],
        "scores": (0.95, 0.91, 0.93, 0.94),
    },
    {
        "title": "Mechanical gating of exciton transport in soft molecular crystals",
        "source": "Science Advances",
        "folder": "Materials Science",
        "summary": "Small reversible strains reconfigure excitonic coupling pathways in a soft molecular lattice, producing a large and repeatable change in transport anisotropy. Ultrafast spectroscopy and molecular dynamics connect the response to a specific packing coordinate.",
        "reason": "Offers a cross-field control principle: mechanically tuning the topology of an energy-migration network.",
        "core": "Reversible strain switches exciton transport by changing one dominant molecular packing coordinate.",
        "innovation": "Links macroscopic mechanical input to ultrafast, direction-selective energy transport.",
        "connection": "The network-topology framing resembles migration-assisted avalanche systems even though the material platform differs.",
        "idea": "Explore whether strain, pressure, or host-lattice distortion could tune migration connectivity in your nanoparticle systems.",
        "labels": ["Cross-field spark", "Frontier"],
        "scores": (0.72, 0.92, 0.97, 0.91),
    },
    {
        "title": "Room-temperature quantum sensing with defect-engineered oxide nanoparticles",
        "source": "ACS Nano",
        "folder": "Nanoscience",
        "summary": "Defect-engineered oxide nanoparticles show optically addressable spin contrast at room temperature. Surface passivation improves stability in aqueous media while preserving magnetic-field sensitivity across a broad particle-size distribution.",
        "reason": "A neighboring sensing platform with useful lessons in defect control, passivation, and single-particle variability.",
        "core": "Surface-passivated oxide nanoparticles retain optically readable spin contrast under ambient and aqueous conditions.",
        "innovation": "Co-optimizes quantum-active defects and colloidal surface chemistry in a scalable nanoscale sensor.",
        "connection": "Its approach to controlling defect populations may inform dopant-environment engineering in optical probes.",
        "idea": "Compare how surface passivation shifts both spin contrast and nonlinear emission thresholds across individual particles.",
        "labels": ["Cross-field spark"],
        "scores": (0.68, 0.88, 0.92, 0.86),
    },
    {
        "title": "Resolving hidden pathways in lanthanide energy migration with multidimensional spectroscopy",
        "source": "Advanced Optical Materials",
        "folder": "Optical Materials",
        "summary": "Multidimensional excitation-emission measurements distinguish parallel migration pathways that appear identical in conventional lifetime traces. A sparse global model recovers pathway-specific rates and uncertainty bounds.",
        "reason": "Direct methodological relevance for identifying hidden transfer pathways in complex lanthanide systems.",
        "core": "Multidimensional spectra separate migration pathways that conventional lifetime analysis conflates.",
        "innovation": "Couples sparse global inference to multidimensional optical measurements with uncertainty estimates.",
        "connection": "This could resolve parameter degeneracy in your multilevel kinetic fits.",
        "idea": "Generate synthetic multidimensional data from your model to identify the minimum experiment needed for parameter identifiability.",
        "labels": ["Field match", "Frontier"],
        "scores": (0.94, 0.89, 0.84, 0.93),
    },
    {
        "title": "Neuromorphic classification using nonlinear photonic nanoparticle ensembles",
        "source": "Nature Communications",
        "folder": "Applied Photonics",
        "summary": "A disordered ensemble of nonlinear nanoparticles performs reservoir-style temporal classification without a trained optical network. Diversity in relaxation times supplies the memory kernel, while a lightweight electronic readout performs the final classification.",
        "reason": "Turns nonlinear response heterogeneity—often treated as noise—into a computational resource.",
        "core": "Relaxation-time diversity in nanoparticle ensembles enables physical reservoir computing.",
        "innovation": "Uses intrinsic optical dynamics as memory instead of engineering a complex photonic circuit.",
        "connection": "Avalanche rise and decay dynamics may provide an unusually tunable nonlinear reservoir.",
        "idea": "Simulate whether an ensemble of avalanche particles with distributed thresholds can classify temporal pulse patterns.",
        "labels": ["Cross-field spark"],
        "scores": (0.70, 0.90, 0.98, 0.90),
    },
    {
        "title": "Bayesian experiment design for time-resolved spectroscopy under photon constraints",
        "source": "Optics Express",
        "folder": "Methods",
        "summary": "An adaptive acquisition strategy selects delay times that maximize expected information gain for competing kinetic models. Simulations and benchmark experiments show comparable parameter precision with substantially fewer detected photons.",
        "reason": "Could reduce acquisition time while improving discrimination between competing kinetic mechanisms.",
        "core": "Adaptive delay selection concentrates scarce photons where kinetic models disagree most.",
        "innovation": "Optimizes measurement timing online using expected information gain.",
        "connection": "Your rate models can provide the competing hypotheses needed by the acquisition policy.",
        "idea": "Use avalanche versus non-avalanche kinetic hypotheses to drive an adaptive low-signal measurement schedule.",
        "labels": ["Field match", "Cross-field spark"],
        "scores": (0.89, 0.87, 0.91, 0.92),
    },
    {
        "title": "Interface phonons reshape energy transfer in core–shell nanostructures",
        "source": "Nano Letters",
        "folder": "Nanoscience",
        "summary": "Isotope-sensitive measurements and atomistic calculations reveal that interface-localized phonons open an additional nonradiative transfer channel in core–shell nanostructures. Shell composition controls both the spectral overlap and coupling strength.",
        "reason": "Highlights an often-hidden interface variable that may alter transfer rates in your core–shell designs.",
        "core": "Localized interface phonons create a shell-tunable nonradiative energy-transfer pathway.",
        "innovation": "Uses isotope substitution to isolate interface-vibrational contributions from bulk phonons.",
        "connection": "It suggests a physical route for unexplained shell-dependent rate changes in kinetic models.",
        "idea": "Add an interface-phonon-mediated rate term and test whether it explains temperature-dependent threshold shifts.",
        "labels": ["Field match", "Frontier"],
        "scores": (0.92, 0.93, 0.86, 0.91),
    },
]


def ensure_demo_data() -> None:
    if latest_dashboard()["run"]:
        return
    now = datetime.now(UTC)
    articles = []
    recommendations = []
    for index, item in enumerate(DEMO_ARTICLES):
        article_id = f"demo-{index + 1}"
        articles.append(
            {
                "id": article_id,
                "title": item["title"],
                "summary": item["summary"],
                "source": item["source"],
                "source_url": "https://www.inoreader.com/",
                "url": "https://www.inoreader.com/",
                "author": "",
                "published_at": (now - timedelta(hours=index * 3 + 1)).isoformat(),
                "folder": item["folder"],
                "summary_quality": 0.92,
                "raw": {"demo": True},
            }
        )
        relevance, novelty, inspiration, confidence = item["scores"]
        recommendations.append(
            {
                "article_id": article_id,
                "relevance_score": relevance,
                "novelty_score": novelty,
                "inspiration_score": inspiration,
                "confidence": confidence,
                "reason": item["reason"],
                "core_finding": item["core"],
                "innovation": item["innovation"],
                "connection": item["connection"],
                "idea": item["idea"],
                "idea_is_speculative": True,
                "labels": item["labels"],
            }
        )
    upsert_articles(articles)
    save_profile("demo-profile", "", DEMO_PROFILE.model_dump())
    refresh_id = create_refresh_run("demo")
    save_recommendations(refresh_id, recommendations)
    complete_refresh_run(
        refresh_id,
        "demo",
        scanned_count=287,
        selected_count=len(recommendations),
        estimated_cost=0.18,
        note="Sample data — connect Inoreader and upload a CV to personalize PaperPulse.",
    )

