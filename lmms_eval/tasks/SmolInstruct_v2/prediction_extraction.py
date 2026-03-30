import re


def llama_forward_synthesis(output_text):
    pattern = "(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"

    output_text = output_text.strip()
    if re.match(pattern + '$', output_text):
        return output_text

    pos = output_text.rfind('Predicted product SMILES:')
    if pos != -1:
        output_text = output_text[pos + 1:].strip()
        return output_text
    
    pos = output_text.rfind(':')
    if pos != -1:
        m = re.match(pattern, output_text[pos + 1:].strip())
        if m is not None:
            output_text = output_text[pos + 1:].strip()
            return output_text[:m.span()[1]]
    
    match_begin = re.match(pattern, output_text)
    if match_begin is not None:
        return output_text[:(match_begin.span())[1]]
    
    return ''


def llama_retrosynthesis(output_text):
    pattern = "(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"
    output_text = output_text.strip()

    m = re.match(pattern, output_text)
    if m is not None:
        span = m.span()
        if span[1] == len(output_text):
            return output_text.strip()
        elif span[1] > 10:
            return output_text[:span[1]]
    
    titles = ('Reactant SMILES:', 'the predicted reactant molecules based on the product SMILES:', 'Here are the predicted reactant molecules in SMILES format:', 'Reactants SMILES:', 'Reactants:', 'the predicted reactants for the given product SMILES:', 'Here are the predicted reactants for the given product molecule:', 'following reactants:', 'Here are the predicted reactants for the given product molecule using the SMILES representation:', 'The predicted reactants for the given product molecule are:', 'the predicted reactants based on the product SMILES:', 'the predicted reactants for the given product:', 'the predicted reactants for the given reaction:')
    found = False
    for title in titles:
        pos = output_text.lower().rfind(title.lower())
        if pos != -1:
            output_text = output_text[pos + len(title):].strip()
            found = True
            break

    if not found:
        for title in ('predicted reactants for each product SMILES:', 'the predicted reactants for each product:', 'the predicted reactants for each product molecule:', 'Here are the predicted reactants for the given product molecules:'):
            pos = output_text.lower().rfind(title.lower())
            if pos != -1:
                output_text = output_text[pos + len(title):].strip()
                pos = output_text.lower().find('product 2:')
                if pos != -1:
                    output_text = output_text[pos + len('product 2:'):].strip().strip('*').strip()
                    found = True
    
    if found:
        match_begin = re.match(pattern, output_text)
        if match_begin is not None:
            return output_text[:(match_begin.span())[1]]
    return ''


def llama_molecule_generation(output_text):
    pattern = "(\[|]|\[[^\]]+]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|;|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])+"
    output_text = output_text.strip()

    pos = output_text.find('\n')
    if pos != -1:
        first_line = output_text[:pos].strip()
    else:
        first_line = output_text
    m = re.match(pattern + '$', first_line)
    if m is not None:
        return first_line
        # span = m.span()
        # if span[1] == len(output_text):
        #     return output_text.strip()
        # elif span[1] > 7:
        #     return output_text[:span[1]]
    
    found = False

    titles = ('SMILES for the second molecule: ', 'the second input:', ' is:', 'molecules:', 'description:', 'you described:', 'SMILES:', 'described in the input:', 'Here is the SMILES representation of the molecule:', 'you provided:', 'the molecule is:')
    for title in titles:
        pos = output_text.lower().find(title.lower())
        if pos != -1:
            output_text = output_text[pos + len(title):].strip()
            found = True
            break
    
    if not found:
        titles= (':',)
        
        for title in titles:
            pos = output_text.lower().rfind(title.lower())
            if pos != -1:
                output_text = output_text[pos + len(title):].strip()
                found = True
                break

    # if not found:
    #     for title in ('predicted reactants for each product SMILES:', 'the predicted reactants for each product:', 'the predicted reactants for each product molecule:', 'Here are the predicted reactants for the given product molecules:'):
    #         pos = output_text.lower().rfind(title.lower())
    #         if pos != -1:
    #             output_text = output_text[pos + len(title):].strip()
    #             pos = output_text.lower().find('product 2:')
    #             if pos != -1:
    #                 output_text = output_text[pos + len('product 2:'):].strip().strip('*').strip()
    #                 found = True
    
    if found:
        match_begin = re.match(pattern, output_text)
        if match_begin is None:
            pos = output_text.find(':')
            output_text = output_text[pos + 1:].strip()
        match_begin = re.match(pattern, output_text)
        if match_begin is not None:
            return output_text[:(match_begin.span())[1]]
    return ''

def llama_molecule_captioning(output_text):
    return output_text.strip()


def llama_name_conversion_i2f(output_text):
    output_text = output_text.strip()
    pos = output_text.rfind(':')
    if pos == -1:
        return output_text
    output_text = output_text[pos + 1:].strip()
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip()

    if len(output_text) > 0 and output_text[-1] == '.':
        output_text = output_text[:-1]
    return output_text


def llama_name_conversion_i2s(output_text):
    return llama_name_conversion_i2f(output_text)


def llama_name_conversion_s2f(output_text):
    return llama_name_conversion_i2s(output_text)


def llama_name_conversion_s2i(output_text):
    output_text = output_text.strip()
    for title in ('IUPAC:', 'IUPAC name:', ':', ' is '):
        pos = output_text.lower().rfind(title.lower())
        if pos != -1:
            output_text = output_text[pos + len(title):].strip()
            break
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip()
    output_text = output_text.strip().strip('.')
    if output_text == 'not provided':
        output_text = ''
    return output_text


def llama_property_prediction_esol(output_text):
    return llama_name_conversion_i2s(output_text)


def llama_property_prediction_lipo(output_text):
    output_text = output_text.strip()
    pos = output_text.rfind(':')
    if pos == -1:
        pos = output_text.rfind('=')
    if pos == -1:
        return output_text
    output_text = output_text[pos + 1:].strip()
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip()
    return output_text


def llama_property_prediction_bbbp(output_text):
    return output_text.strip()


def llama_property_prediction_clintox(output_text):
    return output_text.strip()


def llama_property_prediction_hiv(output_text):
    return output_text.strip()


def llama_property_prediction_sider(output_text):
    return output_text.strip()


def codellama_forward_synthesis(output_text):
    output_text = output_text.strip().strip('.')
    pos = output_text.rfind(':')
    if pos == -1:
        return output_text
    output_text = output_text[pos + 1:].strip().strip('.')
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip().strip('.')
    return output_text


def codellama_retrosynthesis(output_text):
    return codellama_forward_synthesis(output_text)


def codellama_molecule_captioning(output_text):
    return output_text.strip()


def codellama_molecule_generation(output_text):
    return codellama_forward_synthesis(output_text)


def codellama_name_conversion_i2f(output_text):
    return llama_name_conversion_i2s(output_text)


def codellama_name_conversion_i2s(output_text):
    return codellama_forward_synthesis(output_text)


def codellama_name_conversion_s2f(output_text):
    return llama_name_conversion_i2s(output_text)


def codellama_name_conversion_s2i(output_text):
    return llama_name_conversion_s2i(output_text)


def codellama_property_prediction_esol(output_text):
    return output_text.strip()


def codellama_property_prediction_lipo(output_text):
    return output_text.strip()


def codellama_property_prediction_bbbp(output_text):
    return output_text.strip()


def codellama_property_prediction_clintox(output_text):
    return output_text.strip()


def codellama_property_prediction_hiv(output_text):
    return output_text.strip()


def codellama_property_prediction_sider(output_text):
    return output_text.strip()


def mistral_forward_synthesis(output_text):
    output_text = output_text.strip()
    pos = output_text.find('\n')
    if pos == -1:
        return output_text
    else:
        return output_text[:pos].strip()


def mistral_retrosynthesis(output_text):
    return mistral_forward_synthesis(output_text)


def mistral_molecule_captioning(output_text):
    return output_text.strip()


def mistral_molecule_generation(output_text):
    return llama_name_conversion_i2s(output_text)


def mistral_name_conversion_i2f(output_text):
    output_text = output_text.strip()
    output_text = output_text.split('\n')[0].strip()
    return output_text


def mistral_name_conversion_i2s(output_text):
    output_text = output_text.strip()
    pos = output_text.rfind(':')
    if pos != -1:
        output_text = output_text[pos + 1:].strip()
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip()
    return output_text


def mistral_name_conversion_s2f(output_text):
    output_text = output_text.strip()
    output_text = output_text.split('\n')[0].strip()
    return output_text


def mistral_name_conversion_s2i(output_text):
    return llama_name_conversion_s2i(output_text)


def mistral_property_prediction_esol(output_text):
    return mistral_name_conversion_i2s(output_text)


def mistral_property_prediction_lipo(output_text):
    output_text = output_text.strip()
    
    pos = output_text.find('\n')
    if pos != -1:
        first_line = output_text[:pos].strip()
        try:
            _ = float(first_line)
        except:
            pass
        else:
            return first_line

    pos = output_text.rfind(':')
    if pos != -1:
        output_text = output_text[pos + 1:].strip().strip('.')
    else:
        pos = output_text.rfind('is ')
        if pos != -1:
            output_text = output_text[pos + 3:].strip().strip('.')
    
    pos = output_text.find('\n')
    if pos != -1:
        output_text = output_text[:pos].strip()
    return output_text


def mistral_property_prediction_bbbp(output_text):
    return output_text.strip().strip('.')


def mistral_property_prediction_clintox(output_text):
    return output_text.strip().strip('.')


def mistral_property_prediction_hiv(output_text):
    return output_text.strip().strip('.')


def mistral_property_prediction_sider(output_text):
    return output_text.strip().strip('.')


def mol_forward_synthesis(output_text):
    output_text = output_text.replace('<unk>', '').replace('</s>', '').strip()
    return output_text

def mol_retrosynthesis(output_text):
    return mol_forward_synthesis(output_text)


def mol_molecule_captioning(output_text):
    return mol_forward_synthesis(output_text)


def mol_molecule_generation(output_text):
    return mol_forward_synthesis(output_text)


def mol_name_conversion_i2f(output_text):
    return mol_forward_synthesis(output_text)


def mol_name_conversion_i2s(output_text):
    return mol_forward_synthesis(output_text)


def mol_name_conversion_s2f(output_text):
    return mol_forward_synthesis(output_text)


def mol_name_conversion_s2i(output_text):
    return mol_forward_synthesis(output_text)


def mol_property_prediction_esol(output_text):
    output_text = output_text.replace('</s>', '').replace('<unk>', '').strip()
    output_text = output_text.strip('.').strip()
    return output_text


def mol_property_prediction_lipo(output_text):
    output_text = output_text.replace('</s>', '').replace('<unk>', '').strip()
    output_text = output_text.strip('.').strip()
    return output_text


def mol_property_prediction_bbbp(output_text):
    output_text = output_text.strip().lower()
    if output_text[:3] == 'yes':
        return 'Yes'
    elif output_text[:2] == 'no':
        return 'No'
    else:
        return ''


def mol_property_prediction_clintox(output_text):
    return mol_property_prediction_bbbp(output_text)


def mol_property_prediction_hiv(output_text):
    return mol_property_prediction_bbbp(output_text)


def mol_property_prediction_sider(output_text):
    return mol_property_prediction_bbbp(output_text)


def gal_forward_synthesis(output_text):
    title = 'Answer:'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):]
    title = '[START_I_SMILES]'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):]
    pos = output_text.find('[END_I_SMILES]')
    output_text = output_text[:pos]
    output_text = output_text.strip()
    return output_text


def gal_retrosynthesis(output_text):
    return gal_forward_synthesis(output_text)


def gal_molecule_captioning(output_text):
    title = 'Description:'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):].strip()
    title = '\n'
    pos = output_text.find(title)
    if pos == -1:
        return output_text
    output_text = output_text[:pos]
    output_text = output_text.strip()
    return output_text


def gal_molecule_generation(output_text):
    title = 'The SMILES formula of this molecule is [START_I_SMILES]'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):].strip()
    pos = output_text.find('[END_I_SMILES]')
    if pos == -1:
        return output_text
    output_text = output_text[:pos]
    output_text = output_text.strip()
    return output_text


def gal_name_conversion_i2f(output_text):
    title = 'Molecular Formula'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):].strip().strip('*').strip()
    title = '\n'
    pos = output_text.find(title)
    if pos == -1:
        return output_text
    output_text = output_text[:pos]
    output_text = output_text.strip()
    return output_text


def gal_name_conversion_i2s(output_text):
    title = 'Canonical SMILES\n\n[START_SMILES]'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):]
    pos = output_text.find('[END_SMILES]')
    if pos == -1:
        return output_text
    output_text = output_text[:pos]
    output_text = output_text.strip()
    return output_text


def gal_name_conversion_s2f(output_text):
    return gal_name_conversion_i2f(output_text)


def gal_name_conversion_s2i(output_text):
    title = 'The following are chemical properties for '
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):]
    title = '\n'
    pos = output_text.find(title)
    if pos == -1:
        return output_text
    output_text = output_text[:pos].strip().strip('.').strip()
    return output_text


def gal_property_prediction_esol(output_text):
    title = 'Answer:'
    pos = output_text.find(title)
    assert pos != -1
    output_text = output_text[pos + len(title):]
    title = '\n'
    pos = output_text.find(title)
    if pos != -1:
        output_text = output_text[:pos]
    output_text = output_text.strip().replace('</s>', '').strip()
    return output_text


def gal_property_prediction_lipo(output_text):
    return gal_property_prediction_esol(output_text)


def gal_property_prediction_bbbp(output_text):
    return gal_property_prediction_esol(output_text)


def gal_property_prediction_clintox(output_text):
    return gal_property_prediction_esol(output_text)


def gal_property_prediction_hiv(output_text):
    return gal_property_prediction_esol(output_text)


def gal_property_prediction_sider(output_text):
    return gal_property_prediction_esol(output_text)


def chemllm_forward_synthesis(output_text):
    output_text = output_text.replace("</s>", "").strip()
    if output_text[-1] == '.':
        output_text = output_text[:-1]
    return output_text


def chemllm_retrosynthesis(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_molecule_captioning(output_text):
    output_text = output_text.replace("</s>", "").strip()
    return output_text


def chemllm_molecule_generation(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_name_conversion_i2f(output_text):
    output_text = chemllm_forward_synthesis(output_text)
    output_text = output_text.replace(" ", "").replace("_", "")
    return output_text


def chemllm_name_conversion_i2s(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_name_conversion_s2f(output_text):
    return chemllm_name_conversion_i2f(output_text)


def chemllm_name_conversion_s2i(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_property_prediction_esol(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_property_prediction_lipo(output_text):
    return chemllm_forward_synthesis(output_text)


def chemllm_property_prediction_bbbp(output_text):
    output_text = chemllm_forward_synthesis(output_text)
    lower = output_text.lower()
    if lower.startswith('yes') or lower.endswith('yes'):
        return 'Yes'
    elif lower.startswith('no') or lower.endswith('no'):
        return 'No'
    else:
        return ''


def chemllm_property_prediction_clintox(output_text):
    return chemllm_property_prediction_bbbp(output_text)


def chemllm_property_prediction_hiv(output_text):
    return chemllm_property_prediction_bbbp(output_text)


def chemllm_property_prediction_sider(output_text):
    return chemllm_property_prediction_bbbp(output_text)


def extract_pred(sample, model_name, task):
    if model_name == 'mol_trained':
        func_model_name = 'mol'
    else:
        func_model_name = model_name
    func = eval('%s_%s' % (func_model_name, task.lower().replace('-', '_')))
    preds = []
    outputs = sample['output']
    for output in outputs:
        if model_name == 'mol':
            pos = output.find('Response:')
            assert pos != -1
            output = output[pos + len('Response:'):].strip()
        r = func(output)
        preds.append(r)
    return preds