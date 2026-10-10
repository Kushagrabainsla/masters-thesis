# Compute applications (drafts)

Prepared 2026-09-24 from `RESEARCH_JOURNAL.md` (Section 1.7 and Appendix B). All of these are **drafts for Kushagra to review, edit into their own words, and submit**. Nothing has been submitted. Facts, dates and rules were checked on 2026-09-24 and can change, so re-check each program's page before submitting.

| File | What it is | Who submits | Needed by |
|---|---|---|---|
| `advisor_email.md` | Email asking the advisor for a support letter, two questions about the lab, and the NVIDIA co-application | Kushagra | This week |
| `advisor_letter_draft.md` and `.docx` | Support letter for ACCESS Explore and NAIRR. The advisor must edit and sign it on letterhead as a PDF | Advisor | Before either submission |
| `nairr_startup_proposal.md` and `.pdf` | NAIRR Pilot Start-Up proposal in NAIRR's required sections A to D, about 2.2 pages of the 3 allowed. Requests TACC Horizon (Blackwell), with a Vista (H200) fallback, 500 GPU-hours and 300 GB | Kushagra | Now. Decision within 3 weeks |
| `access_explore_request.md` | ACCESS Explore fields and project summary for Delta H200. The amount needs the ACCESS exchange calculator | Kushagra | Now |
| `nvidia_academic_grant_outline.md` | Outline for NVIDIA's inference-systems track. Faculty must be the applicant | Advisor | Window opens 2026-10-01 |

## Order

1. Send the advisor email with the letter draft and the two request drafts attached.
2. Read the ACCESS exchange calculator and fill in the ACCESS amount. Submit ACCESS Explore.
3. Open the NAIRR resource catalog. If TACC Horizon is selectable for a Start-Up, keep the Horizon text. If not, switch the resource to TACC Vista and change the affected sentences (marked in the proposal's request table and section 6).
4. Submit NAIRR Start-Up with the signed letter.
5. Prepare the NVIDIA proposal once the template is available.

## NAIRR format rules (official instructions, fetched 2026-09-24)

- At most 3 pages as a PDF. Times New Roman 11pt or larger (tables too), 1-inch margins, at most 6 lines per inch.
- Sections: A Scientific/Technical Goal (about one page), B Estimate of Compute, Storage and Other Resources (about one page), C Support Needs (C1 collaboration and support, C2 compliance and access restrictions), D Team and Team Preparedness (qualifications, readiness, citizenship status for federal facility access).
- "Resource estimates should be based on experience in prior work or benchmark runs." Ours is a planning estimate because there is no local GPU, and the proposal says so.
- "The justification must appear both in the proposal text and be clearly reflected in the online form entries." So the resource, GPU-hours and storage on the online form must match the proposal: 500 GPU-hours, 300 GB.
- Institutional email required. Attach a CV or biosketch and the advisor's support letter.

## Things to check before submitting

- The NAIRR draft says the PI has "prior experience in GPU checkpoint and restore systems". That comes from the LinkedIn internship. Confirm you are allowed to mention it.
- The GPU-hour estimate (342, request 500) is a planning estimate, not a measurement. Keep that wording.
- Section D has a placeholder for citizenship or residency status. Fill it in yourself.
- A labmate's NAIRR proposal (`grants/reference/NAIRR_Proposal.pdf`, ignored by git) was used for structure and tone only. It is their work, so do not copy sentences or commit it.
- The proposal quotes numbers from arXiv 2608.13756, 2609.00363, DriftBench and two vLLM issues. All were checked on 2026-09-23. Re-check the vLLM issue states before submitting, since they are open and change.
- Use the friend's proposal for structure and tone only. Every sentence you submit should be your own.

## Rebuilding the PDF

`./build_pdf.sh nairr_startup_proposal` renders the Markdown to PDF with headless Chrome and `style.css`, then check that it is still 3 pages.
