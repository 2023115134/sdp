GitHub Copilot Prompt — Secure and Covert Communication Demo UI
Act as a senior full-stack developer. Examine my existing project repository and build a professional, interactive web UI for my project:
**“An LLM-Based Framework for Secure and Covert Communication over Public Message Channels.”**
1. First, inspect the existing project
Before changing any files:
Inspect the complete repository structure.
Identify the existing programming language, framework, entry points, backend modules, encryption functions, LLM integration, and sender/receiver implementation.
Find the actual functions used for PBKDF2, AES-256-GCM, h4 character mapping, SHAKE-128 position generation, EmbedderLLM, payload extraction, and authenticated decryption.
Determine how the current demo is launched and what inputs and outputs each module expects.
Reuse the existing implementation instead of rewriting working cryptographic or LLM code.
If a component is not implemented, clearly mark it as incomplete rather than pretending it works.
Create a short implementation plan, then proceed with the implementation.
2. Build a professional demo dashboard
Create a modern, clean, responsive interface suitable for a college project review and live demonstration.
Use the framework already present in the repository. If there is no frontend framework, choose a lightweight solution compatible with the existing backend.
The UI should have these sections:
Dashboard / Overview
Sender Pipeline
Intermediate Outputs
Public Message Channel
Receiver Pipeline
Execution Logs and Verification Results
Use cards, step indicators, expandable technical details, status labels, copy buttons, and readable code/text panels. The interface should look polished and academic, not like a generic admin dashboard.
3. Sender-side interface
Provide the following inputs:
Secret plaintext message
Key-establishment mode: PSK or ECDHE, only if supported by the current implementation
Password or pre-shared secret, where applicable
Carrier text topic
LLM settings, such as temperature and top-k, if supported
Run Demo button
Reset button
Never log or display passwords, private keys, or secret key bytes in the UI or console.
When the user clicks Run Demo, execute the real sender pipeline and show its intermediate outputs step by step.
4. Show all intermediate outputs
This is the most important requirement.
Display each intermediate stage separately, with its actual output, a short explanation, and a status indicating success, failure, or incomplete implementation.
Stage A — Key establishment
Show:
Selected mode: PSK or ECDHE
A human-readable explanation of the key-establishment process
Whether the shared secret or key establishment completed successfully
Public-key exchange status in ECDHE mode, if applicable
Do not expose shared secrets, private keys, passwords, or derived key bytes.
Stage B — PBKDF2 key derivation
Show:
Algorithm and configured iteration count
Salt representation, only if appropriate for the existing protocol
Derived key length: 64 bytes, if this matches the actual implementation
DK1 length: 32 bytes
DK2 length: 32 bytes
A clear explanation of DK1 and DK2 purposes
Show key lengths and status, not the secret key values.
Stage C — AES-256-GCM encryption
Show:
Original plaintext in a clearly marked panel
Encryption algorithm
Nonce representation, if available
Additional Authenticated Data (AAD), if used
Ciphertext in hexadecimal
Authentication tag in hexadecimal
Ciphertext length in bytes
Authentication tag length in bytes
For the described design, the tag is normally 16 bytes. Confirm this against the actual implementation.
Do not replace real encryption with a mock or simulated result.
Stage D — Payload construction
Show:
The actual payload order used by the implementation
For the documented algorithm, this is tag concatenated with ciphertext: tag || ciphertext
Combined payload in hexadecimal
Payload length in bytes and hexadecimal characters
Make the distinction between ciphertext and authentication tag visually clear.
Stage E — h4 character mapping
Show:
Input hexadecimal payload
Output of the actual h4 mapping function
Character count before and after mapping
A short explanation of how the inverse mapping reconstructs the original representation
Do not invent the mapping. Read and use the mapping function already present in the repository.
Stage F — SHAKE-128 target-position generation
Show:
Algorithm: SHAKE-128
The number of target positions generated
The actual target-position sequence, preferably in a scrollable table
The relationship between each mapped character and its target position
A short explanation that DK2 is used to regenerate the positions at the receiver
Use the actual position-generation function and parameters. Do not invent positions or substitute random values.
Avoid exposing DK2 or any secret key material.
Stage G — EmbedderLLM carrier generation
Show:
Selected topic
Current generation parameters
Number of hidden characters to embed
Target position and corresponding hidden character, in a table if the implementation makes these available
Candidate token attempts or retry information, if exposed by the actual algorithm
Rollback/retry events, if available
Generated carrier text
Embedding completion status
The generated carrier text must come from the real LLM integration.
If the embedding algorithm does not expose internal candidate tokens or retries, show only the data it actually returns. Do not fabricate detailed execution traces.
Stage H — Final sender output
Display:
Final carrier text in a large, readable text box
Copy Carrier Text button
Download Carrier Text button, if practical
Total carrier text length
Overall sender pipeline status
Include a short message explaining that the carrier text is intended to conceal the encrypted payload within natural-language text. Do not claim that it is guaranteed to be undetectable.
5. Public message channel simulation
Create a panel between sender and receiver representing the public message channel.
Display only the carrier text that is intended to be transmitted.
Provide a button such as “Deliver Carrier Text to Receiver.”
Pass the actual carrier text to the receiver pipeline.
Show the transmission status.
Clearly distinguish the public carrier-text channel from the TCP key-exchange connection used by the ECDHE demo, if that connection exists in the repository.
Do not silently transfer secret plaintext, derived keys, or hidden payloads through the simulated public channel.
6. Receiver-side interface
When the receiver receives the carrier text, display each actual processing stage.
Stage I — Received carrier text
Show the exact carrier text received by the receiver.
Allow a user to paste or edit carrier text for testing, where supported.
Stage J — Key derivation
Show the receiver's key-establishment mode, PBKDF2 parameters, and derived-key lengths, without displaying secret values.
The receiver must use the correct shared secret/password and protocol parameters.
Stage K — Position regeneration and extraction
Show:
Regenerated target positions
Extracted characters
Extracted payload representation
Extraction status
Use the real receiver-side extraction logic.
Stage L — Inverse h4 mapping and payload reconstruction
Show:
Extracted mapped characters
Output of the inverse mapping
Reconstructed authentication tag
Reconstructed ciphertext
Lengths of the recovered components
Use the same tag/ciphertext ordering as the actual sender implementation.
Stage M — Authenticated decryption
Show:
AES-256-GCM verification status
Authentication success or failure
Recovered plaintext only after authentication succeeds
A clear error message if verification fails
Never show plaintext as successfully recovered before the AEAD authentication operation has succeeded.
If authentication fails, do not return unauthenticated plaintext.
7. Intermediate-output visualisation
Create a pipeline progress component:
Key Establishment → PBKDF2 → AES-GCM → Payload Construction → h4 Mapping → SHAKE-128 Positions → EmbedderLLM → Carrier Text → Extraction → Inverse Mapping → AEAD Verification → Plaintext
Each step should have:
Pending, running, success, failed, or not implemented status
Actual intermediate data, where appropriate
An expandable details section
A concise technical explanation
Add a “Run Step by Step” mode if practical, allowing me to execute one stage at a time for my review presentation.
Also include a “Run Complete Demo” option.
Do not display success for a stage unless the corresponding operation actually succeeded.
8. Backend integration requirements
Integrate the UI with the existing backend through suitable local API endpoints or the project's existing interface.
Keep cryptographic operations on the backend where possible.
Reuse existing functions instead of implementing duplicate algorithms.
Do not put passwords, private keys, or other secrets into frontend logs or browser storage.
Validate inputs and handle errors clearly.
Keep the sender and receiver logic consistent with the actual project.
Preserve the current terminal-based demo if possible.
Do not replace the LLM with hardcoded carrier text.
Do not replace encryption with Base64, simple character substitutions, or dummy ciphertext.
Do not invent outputs for unfinished modules.
Ensure the sender and receiver use matching parameters, including the nonce, AAD, salt, and iteration count where applicable.
Handle authentication failure safely.
9. Demo testing
Add or update tests for the following cases, according to the existing implementation:
Successful encryption, embedding, extraction, authentication, and decryption.
Empty and invalid input.
Wrong password or mismatched keys.
Modified extracted payload causing authentication failure.
Mismatched nonce or AAD causing authentication failure, where applicable.
Carrier text that does not contain a valid embedded payload.
LLM generation failure or unavailable model.
Incorrect or out-of-range target positions.
Do not claim tests passed unless you execute them and observe the result.
10. Documentation and final response
After implementing the UI:
Tell me which files you created or modified.
Explain how to start the backend and frontend in separate terminals.
Provide the exact commands for the existing operating system and environment.
Tell me the local URL to open, based on the actual server configuration.
Explain the demo flow from sender input to receiver plaintext recovery.
Explain which intermediate outputs are real and which features remain incomplete.
List any dependencies that must be installed.
Run the available tests and report the actual results.
Do not delete or restructure existing working modules unnecessarily.
Priority: Build a functioning demo UI integrated with my existing code. The main purpose is to show the panel every important intermediate output in the sender and receiver pipelines, not merely to create a static visual mockup.
